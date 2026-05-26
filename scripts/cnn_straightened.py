import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from mode_csv import read_mode_csv_entries
from nova_mode_loader import load_mode_from_nova
from mode_transform import resample_r, straighten_mode_window
from paths import NOVA_TRAIN_CSV
from cnn_infer_common import CHECKPOINT_VERSION, build_preprocess_metadata
from torch_runtime import print_torch_device_report, select_torch_device


# =========================
# 0) Plug in your loader
# =========================
"""
    Now in nova_mode_loader.py
"""
# =========================
# 1) CSV utilities
# =========================
def read_train_csv(csv_path: str, data_root: str | None = None) -> List[Dict[str, Any]]:
    items = []
    for p, raw_label in read_mode_csv_entries(csv_path, data_root=data_root):
        lab = (raw_label or "").strip().lower()
        if lab not in ("good", "bad"):
            raise ValueError(f"Bad label '{lab}' in {csv_path} for path {p}")
        y = 1 if lab == "good" else 0
        items.append({"path": p, "label": y})
    return items


def train_test_split_stratified(items: List[Dict[str, Any]], test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    items = list(items)
    y = np.array([it["label"] for it in items])

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)

    n0_test = max(1, int(len(idx0) * test_frac))
    n1_test = max(1, int(len(idx1) * test_frac))

    test_idx = np.concatenate([idx0[:n0_test], idx1[:n1_test]])
    train_idx = np.concatenate([idx0[n0_test:], idx1[n1_test:]])

    rng.shuffle(train_idx); rng.shuffle(test_idx)

    train_items = [items[i] for i in train_idx]
    test_items = [items[i] for i in test_idx]
    return train_items, test_items


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 2) Dataset
# =========================

class NovaModeDataset(Dataset):
    def __init__(
        self,
        items,
        normalize="robust",
        M=8,
        center_power=2.0,
        median_k=3,
        max_step=2,
        R_target=201,
        cache_data=False,
    ):
        self.items = items
        self.normalize = normalize
        self.M = M
        self.center_power = center_power
        self.median_k = median_k
        self.max_step = max_step
        self.R_target = R_target
        self.cached_samples = None
        if cache_data:
            t0 = time.perf_counter()
            self.cached_samples = [self._load_sample(i) for i in range(len(self.items))]
            print(
                f"Cached {len(self.cached_samples)} straightened CNN samples in "
                f"{time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    def _normalize(self, x):
        if self.normalize == "none":
            return x
        if self.normalize == "robust":
            med = float(np.median(x))
            mad = float(np.median(np.abs(x - med)))
            if mad < 1e-3:
                return x - med
            return (x - med) / (mad + 1e-8)
        if self.normalize == "maxabs":
            s = float(np.max(np.abs(x))) + 1e-8
            return x / s
        if self.normalize == "standard":
            mu = float(np.mean(x))
            sig = float(np.std(x)) + 1e-8
            return (x - mu) / sig
        raise ValueError("normalize must be none|standard|robust|maxabs")

    def __len__(self):
        return len(self.items)

    def _load_sample(self, idx):
        it = self.items[idx]
        mode, omega, gamma_d, ntor = load_mode_from_nova(it["path"])

        mode = resample_r(mode, R_target=self.R_target)
        x2, mc, mc_int = straighten_mode_window(
            mode,
            M=self.M,
            center_power=self.center_power,
            median_k=self.median_k,
            max_step=self.max_step,
        )  # (25, n_r)

        x = x2[None, :, :]  # (1,25,n_r)
        x = self._normalize(x)

        y = torch.tensor(it["label"], dtype=torch.long)
        return torch.from_numpy(x), y, it["path"]

    def __getitem__(self, idx):
        if self.cached_samples is not None:
            return self.cached_samples[idx]
        return self._load_sample(idx)

# =========================
# 3) Small CNN: size-agnostic via AdaptiveAvgPool2d
# =========================
class SmallCNN(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 1)  # logits
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x).squeeze(1)
        return x


# =========================
# 4) Train / Eval loops
# =========================
def train_epoch(model, loader, opt, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()

    total = 0.0
    n = 0
    for x, y, _paths in loader:
        x = x.to(device)
        y = y.float().to(device)

        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_model(model, loader, device, thr: float = 0.5):
    model.eval()
    probs_all = []
    y_all = []
    paths_all = []

    for x, y, paths in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        probs_all.append(probs)
        y_all.append(y.numpy())
        paths_all += paths

    probs_all = np.concatenate(probs_all)
    y_all = np.concatenate(y_all)
    pred = (probs_all >= thr).astype(int)
    acc = float(np.mean(pred == y_all))
    return acc, probs_all, y_all, paths_all


# =========================
# 5) Main
# =========================
@dataclass
class Config:
    train_csv: str = str(NOVA_TRAIN_CSV)
    data_dir: str | None = os.environ.get("NOVA_DATA")
    test_frac: float = 0.2
    seed: int = 42
    batch_size: int = 32
    epochs: int = 80
    lr: float = 1e-2
    normalize: str = "maxabs" # "robust"  # "none" is OK too since max=1
    eval_threshold: float = 0.5
    model_out: str = "nova_cnn.pt"
    M: int = 8
    center_power: float = 2.0
    median_k: int = 3
    max_step: int = 2
    R_target: int = 201
    device: str | None = os.environ.get("NOVA_TORCH_DEVICE")
    cache_data: bool = False
    refit_full_before_save: bool = False


def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description=(
            "Train the straightened NOVA CNN on ridge-aligned mode windows. "
            "Relative paths in the training CSV are resolved with --data_dir or $NOVA_DATA."
        )
    )
    ap.add_argument("--train_csv", default=str(NOVA_TRAIN_CSV), help="Training CSV with mode paths and labels")
    ap.add_argument(
        "--data_dir",
        default=os.environ.get("NOVA_DATA"),
        help="Data directory used to resolve relative mode paths in --train_csv (default: $NOVA_DATA)",
    )
    ap.add_argument("--model_out", default="nova_cnn.pt", help="Output checkpoint path")
    ap.add_argument("--test_frac", type=float, default=0.2, help="Stratified test fraction")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split and training")
    ap.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    ap.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    ap.add_argument("--lr", type=float, default=1e-2, help="Initial Adam learning rate")
    ap.add_argument(
        "--normalize",
        choices=["none", "standard", "robust", "maxabs"],
        default="maxabs",
        help="Per-mode image normalization",
    )
    ap.add_argument("--eval_threshold", type=float, default=0.5, help="Probability threshold used for metrics")
    ap.add_argument("--M", type=int, default=8, help="Half-width of the straightened harmonic window")
    ap.add_argument("--center_power", type=float, default=2.0, help="Power used to estimate ridge center")
    ap.add_argument("--median_k", type=int, default=3, help="Median-filter width for ridge center")
    ap.add_argument("--max_step", type=int, default=2, help="Maximum ridge-center step between radial samples")
    ap.add_argument("--R_target", type=int, default=201, help="Radial grid size after interpolation")
    ap.add_argument(
        "--device",
        default=os.environ.get("NOVA_TORCH_DEVICE"),
        help="Torch device, e.g. cpu, cuda, cuda:0 (default: $NOVA_TORCH_DEVICE or auto)",
    )
    ap.add_argument(
        "--cache_data",
        action="store_true",
        help="Preload preprocessed mode tensors into RAM once instead of rereading files every epoch",
    )
    ap.add_argument(
        "--refit_full_before_save",
        action="store_true",
        help=(
            "After selecting best_epoch on the held-out split, train a fresh "
            "model on the full labeled CSV for best_epoch epochs and save that "
            "full-data model. Held-out metrics are still printed from the "
            "initial split model."
        ),
    )
    return Config(**vars(ap.parse_args()))


def main():
    cfg = parse_args()

    seed_everything(cfg.seed)

    items = read_train_csv(cfg.train_csv, data_root=cfg.data_dir)
    train_items, test_items = train_test_split_stratified(items, cfg.test_frac, cfg.seed)

    print(f"Training CSV: {cfg.train_csv}")
    print(f"Data dir: {cfg.data_dir or '$NOVA_DATA'}")
    print(f"Total modes: {len(items)} | Train: {len(train_items)} | Test: {len(test_items)}")
    print(
        "Straightened preprocessing: "
        f"R_target={cfg.R_target}, M={cfg.M}, center_power={cfg.center_power}, "
        f"median_k={cfg.median_k}, max_step={cfg.max_step}"
    )

    train_ds = NovaModeDataset(
        train_items,
        normalize=cfg.normalize,
        M=cfg.M,
        center_power=cfg.center_power,
        median_k=cfg.median_k,
        max_step=cfg.max_step,
        R_target=cfg.R_target,
        cache_data=cfg.cache_data,
    )
    test_ds  = NovaModeDataset(
        test_items,
        normalize=cfg.normalize,
        M=cfg.M,
        center_power=cfg.center_power,
        median_k=cfg.median_k,
        max_step=cfg.max_step,
        R_target=cfg.R_target,
        cache_data=cfg.cache_data,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=0)

    device = select_torch_device(cfg.device)
    print_torch_device_report(device)

    model = SmallCNN(in_ch=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )

    best_acc = -1.0
    best_epoch = 0
    best_state = None

    for ep in range(1, cfg.epochs + 1):
        loss = train_epoch(model, train_loader, opt, device)
        acc, probs, y_true, _ = eval_model(model, test_loader, device, thr=cfg.eval_threshold)
        sched.step(acc)

        if acc > best_acc:
            best_acc = acc
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 5 == 0:
            lr = opt.param_groups[0]["lr"]
            print(f"Epoch {ep:3d}/{cfg.epochs} | loss={loss:.4f} | test_acc={acc:.4f} | lr={lr:.2e}")

    print(f"Best test acc: {best_acc:.4f} (epoch {best_epoch})")
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics
    acc, probs, y_true, paths = eval_model(model, test_loader, device, thr=cfg.eval_threshold)
    thr = cfg.eval_threshold
    y_pred = (probs >= thr).astype(int)
    #for thr in np.linspace(0.4, 0.75, 8):     # sweep thresholds to check nmber of FPs
        #y_pred = (probs >= thr).astype(int)
        #tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        #print(f"thr={thr:.2f}  FP={fp}  FN={fn}  TP={tp}")

    # Check FP cases
    wrong = np.where(y_pred != y_true)[0]
    print("\nMisclassified modes:")
    print("Path,  true_label,  pred_label,  p_good")
    for i in wrong:
        true_lab = "good" if y_true[i] == 1 else "bad"
        pred_lab = "good" if y_pred[i] == 1 else "bad"
        print(f"{paths[i]}, {true_lab}, {pred_lab}, {probs[i]:.3f}")

    print("\nGood modes p_good range:",
          probs[y_true==1].min(), probs[y_true==1].max())
    print("Bad modes p_good range:",
          probs[y_true==0].min(), probs[y_true==0].max())

    print("mean p_good | true good:", probs[y_true==1].mean())
    print("mean p_good | true bad :", probs[y_true==0].mean())

    print("\nConfusion matrix (rows=actual [bad,good], cols=pred [bad,good]):")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["bad", "good"]))

    preprocess_meta = build_preprocess_metadata(
        R_target=cfg.R_target,
        M=cfg.M,
        center_power=cfg.center_power,
        median_k=cfg.median_k,
        max_step=cfg.max_step,
    )

    model_to_save = model
    saved_training_scope = "train_split"
    final_train_epochs = 0
    final_train_size = len(train_items)

    if cfg.refit_full_before_save:
        if best_epoch <= 0:
            raise RuntimeError("Cannot refit on the full set because no best epoch was selected.")

        print(
            f"\nRefitting final straightened CNN on all {len(items)} labeled modes "
            f"for {best_epoch} epochs before saving.",
            flush=True,
        )
        full_ds = NovaModeDataset(
            items,
            normalize=cfg.normalize,
            M=cfg.M,
            center_power=cfg.center_power,
            median_k=cfg.median_k,
            max_step=cfg.max_step,
            R_target=cfg.R_target,
            cache_data=cfg.cache_data,
        )
        seed_everything(cfg.seed)
        full_loader = DataLoader(full_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        final_model = SmallCNN(in_ch=1).to(device)
        final_opt = torch.optim.Adam(final_model.parameters(), lr=cfg.lr)

        for ep in range(1, best_epoch + 1):
            epoch_t0 = time.perf_counter()
            loss = train_epoch(final_model, full_loader, final_opt, device)
            if ep == 1 or ep == best_epoch or ep % 5 == 0:
                elapsed = time.perf_counter() - epoch_t0
                print(
                    f"Full-fit epoch {ep:3d}/{best_epoch} | loss={loss:.4f} | "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

        model_to_save = final_model
        saved_training_scope = "full_csv_refit"
        final_train_epochs = best_epoch
        final_train_size = len(items)

    torch.save({
        "model_state_dict": model_to_save.state_dict(),
        "seed": cfg.seed,
        "normalize": cfg.normalize,
        "threshold": thr,
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "test_frac": cfg.test_frac,
        "initial_train_size": len(train_items),
        "test_size": len(test_items),
        "refit_full_before_save": cfg.refit_full_before_save,
        "saved_training_scope": saved_training_scope,
        "final_train_epochs": final_train_epochs,
        "final_train_size": final_train_size,
        "model_type": "cnn_straightened",
        "checkpoint_version": CHECKPOINT_VERSION,
        "preprocess": preprocess_meta,
        **preprocess_meta,
    }, cfg.model_out)
    print(f"\nSaved CNN to {cfg.model_out} ({saved_training_scope})")


if __name__ == "__main__":
    main()
