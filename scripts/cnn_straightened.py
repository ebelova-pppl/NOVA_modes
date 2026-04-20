import random
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


# =========================
# 0) Plug in your loader
# =========================
"""
    Now in nova_mode_loader.py
"""
# =========================
# 1) CSV utilities
# =========================
def read_train_csv(csv_path: str) -> List[Dict[str, Any]]:
    items = []
    for p, raw_label in read_mode_csv_entries(csv_path):
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
    ):
        self.items = items
        self.normalize = normalize
        self.M = M
        self.center_power = center_power
        self.median_k = median_k
        self.max_step = max_step
        self.R_target = R_target

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

    def __getitem__(self, idx):
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
def eval_model(model, loader, device):
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
    pred = (probs_all >= 0.5).astype(int)
    acc = float(np.mean(pred == y_all))
    return acc, probs_all, y_all, paths_all


# =========================
# 5) Main
# =========================
@dataclass
class Config:
    train_csv: str = str(NOVA_TRAIN_CSV)
    test_frac: float = 0.2
    seed: int = 42
    batch_size: int = 32
    epochs: int = 80
    lr: float = 1e-2
    normalize: str = "maxabs" # "robust"  # "none" is OK too since max=1
    eval_threshold: float = 0.55
    model_out: str = "nova_cnn.pt"
    M: int = 8
    center_power: float = 2.0
    median_k: int = 3
    max_step: int = 2
    R_target: int = 201


def main():
    cfg = Config()

    seed_everything(cfg.seed)

    items = read_train_csv(cfg.train_csv)
    train_items, test_items = train_test_split_stratified(items, cfg.test_frac, cfg.seed)

    print(f"Total modes: {len(items)} | Train: {len(train_items)} | Test: {len(test_items)}")

    train_ds = NovaModeDataset(
        train_items,
        normalize=cfg.normalize,
        M=cfg.M,
        center_power=cfg.center_power,
        median_k=cfg.median_k,
        max_step=cfg.max_step,
        R_target=cfg.R_target,
    )
    test_ds  = NovaModeDataset(
        test_items,
        normalize=cfg.normalize,
        M=cfg.M,
        center_power=cfg.center_power,
        median_k=cfg.median_k,
        max_step=cfg.max_step,
        R_target=cfg.R_target,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print("Device:", device)

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
        acc, probs, y_true, _ = eval_model(model, test_loader, device)
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
    acc, probs, y_true, paths = eval_model(model, test_loader, device)
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

    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "seed": cfg.seed,
        "normalize": cfg.normalize,
        "threshold": thr,
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "model_type": "cnn_straightened",
        "checkpoint_version": CHECKPOINT_VERSION,
        "preprocess": preprocess_meta,
        **preprocess_meta,
    }, cfg.model_out)
    print(f"\nSaved CNN to {cfg.model_out}")


if __name__ == "__main__":
    main()
