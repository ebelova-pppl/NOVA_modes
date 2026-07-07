#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

from cnn_infer_common import CHECKPOINT_VERSION, build_raw_preprocess_metadata
from mode_csv import read_mode_csv_entries
from mode_transform import resample_r
from nova_mode_loader import load_mode_from_nova
from torch_runtime import print_torch_device_report, select_torch_device


COLLAPSE_CHECK_START_EPOCH = 5
COLLAPSE_CLASS_FRACTION = 0.02
COLLAPSE_PROB_STD = 1e-3


def default_train_csv() -> str:
    env_value = os.environ.get("NOVA_TRAIN_CSV")
    if env_value:
        return env_value
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / "training_labels" / "tae_like_train.csv")


def read_train_csv(csv_path: str, data_root: str | None = None) -> list[dict[str, Any]]:
    items = []
    for p, raw_label in read_mode_csv_entries(csv_path, data_root=data_root):
        lab = (raw_label or "").strip().lower()
        if lab not in ("good", "bad"):
            raise ValueError(f"Bad label '{lab}' in {csv_path} for path {p}")
        y = 1 if lab == "good" else 0
        items.append({"path": p, "label": y})
    return items


def train_test_split_stratified(items: list[dict[str, Any]], test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    items = list(items)
    y = np.array([it["label"] for it in items])

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_test = max(1, int(len(idx0) * test_frac))
    n1_test = max(1, int(len(idx1) * test_frac))

    test_idx = np.concatenate([idx0[:n0_test], idx1[:n1_test]])
    train_idx = np.concatenate([idx0[n0_test:], idx1[n1_test:]])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

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


def pad_or_crop(mode: np.ndarray, Mt: int = 54, Rt: int = 201) -> np.ndarray:
    mode = np.asarray(mode, dtype=np.float32)
    n_m, n_r = mode.shape
    out = np.zeros((Mt, Rt), dtype=np.float32)
    mmin = min(n_m, Mt)
    rmin = min(n_r, Rt)
    out[:mmin, :rmin] = mode[:mmin, :rmin]
    return out


class NovaModeDataset(Dataset):
    def __init__(
        self,
        items,
        normalize: str = "robust",
        M_target: int = 54,
        R_target: int = 201,
        cache_data: bool = False,
    ):
        self.items = items
        self.normalize = normalize
        self.M_target = M_target
        self.R_target = R_target
        self.cached_samples = None
        if cache_data:
            t0 = time.perf_counter()
            self.cached_samples = [self._load_sample(i) for i in range(len(self.items))]
            print(
                f"Cached {len(self.cached_samples)} raw CNN samples in "
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
        mode = pad_or_crop(mode, Mt=self.M_target, Rt=self.R_target)
        x = mode[None, :, :]
        x = self._normalize(x)

        y = torch.tensor(it["label"], dtype=torch.long)
        return torch.from_numpy(x), y, it["path"]

    def __getitem__(self, idx):
        if self.cached_samples is not None:
            return self.cached_samples[idx]
        return self._load_sample(idx)


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
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x).squeeze(1)
        return x


def train_epoch(
    model,
    loader,
    opt,
    device,
    pos_weight: float | None = None,
    *,
    batch_scheduler=None,
    grad_clip_norm: float | None = None,
):
    model.train()
    pos_weight_tensor = None
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    total = 0.0
    n = 0
    for x, y, _paths in loader:
        x = x.to(device)
        y = y.float().to(device)

        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        opt.step()
        if batch_scheduler is not None:
            batch_scheduler.step()

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


@dataclass
class Config:
    train_csv: str
    data_dir: str | None = None
    test_frac: float = 0.2
    seed: int = 42
    batch_size: int = 32
    epochs: int = 80
    lr: float = 2e-2
    pos_weight: str | None = None
    normalize: str = "robust"
    eval_threshold: float = 0.5
    model_out: str = "nova_cnn.pt"
    M_target: int = 54
    R_target: int = 201
    device: str | None = None
    cache_data: bool = False
    refit_full_before_save: bool = False
    onecycle_div_factor: float = 20.0
    onecycle_final_div_factor: float = 100.0
    onecycle_pct_start: float = 0.1
    grad_clip_norm: float | None = 1.0


@dataclass(frozen=True)
class PredictionHealth:
    n_samples: int
    n_true_good: int
    n_predicted_good: int
    predicted_good_fraction: float
    true_good_fraction: float
    prob_mean: float
    prob_std: float
    prob_min: float
    prob_max: float
    collapse_reasons: tuple[str, ...]

    @property
    def collapse_detected(self) -> bool:
        return bool(self.collapse_reasons)


def summarize_prediction_health(
    probs: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
) -> PredictionHealth:
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if probs.size == 0 or y_true.size == 0:
        raise ValueError("Cannot check prediction health with no samples.")
    if probs.shape != y_true.shape:
        raise ValueError(
            "Prediction-health arrays must have matching shapes, "
            f"got probs={probs.shape} and labels={y_true.shape}."
        )

    predicted_good_fraction = float(np.mean(probs >= threshold))
    true_good_fraction = float(np.mean(y_true == 1))
    n_samples = int(probs.size)
    n_predicted_good = int(np.count_nonzero(probs >= threshold))
    n_true_good = int(np.count_nonzero(y_true == 1))
    n_true_bad = n_samples - n_true_good
    prob_std = float(np.std(probs))
    reasons = []

    if n_true_good > 0 and n_predicted_good == 0:
        reasons.append("zero predicted GOOD modes")
    elif (
        true_good_fraction >= COLLAPSE_CLASS_FRACTION
        and predicted_good_fraction < COLLAPSE_CLASS_FRACTION
    ):
        reasons.append("near-all-bad predictions")

    if n_true_bad > 0 and n_predicted_good == n_samples:
        reasons.append("zero predicted BAD modes")
    elif (
        (1.0 - true_good_fraction) >= COLLAPSE_CLASS_FRACTION
        and predicted_good_fraction > 1.0 - COLLAPSE_CLASS_FRACTION
    ):
        reasons.append("near-all-good predictions")
    if prob_std < COLLAPSE_PROB_STD:
        reasons.append("near-constant probabilities")

    return PredictionHealth(
        n_samples=n_samples,
        n_true_good=n_true_good,
        n_predicted_good=n_predicted_good,
        predicted_good_fraction=predicted_good_fraction,
        true_good_fraction=true_good_fraction,
        prob_mean=float(np.mean(probs)),
        prob_std=prob_std,
        prob_min=float(np.min(probs)),
        prob_max=float(np.max(probs)),
        collapse_reasons=tuple(reasons),
    )


def report_prediction_health(
    stage: str,
    epoch: int,
    health: PredictionHealth,
) -> None:
    if epoch < COLLAPSE_CHECK_START_EPOCH or not health.collapse_detected:
        return

    print(
        f"WARNING: {stage.lower()} prediction collapse at epoch {epoch}: "
        f"{'; '.join(health.collapse_reasons)}. "
        f"predicted_good={health.n_predicted_good}/{health.n_samples} "
        f"({health.predicted_good_fraction:.4f}), "
        f"true_good={health.n_true_good}/{health.n_samples} "
        f"({health.true_good_fraction:.4f}), "
        f"p_good mean/std={health.prob_mean:.4f}/{health.prob_std:.4f}, "
        f"range=[{health.prob_min:.4f}, {health.prob_max:.4f}]. "
        "The model may be stalled.",
        flush=True,
    )


def parse_optional_positive_float(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"none", "off", "false", "0"}:
        return None

    try:
        parsed = float(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a positive number or one of: none, off, 0"
        ) from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            "expected a positive number or one of: none, off, 0"
        )
    return parsed


def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description=(
            "Train the raw NOVA CNN on padded/cropped (m,r) mode arrays. "
            "Relative paths in the training CSV are resolved with --data_dir or $NOVA_DATA."
        )
    )
    ap.add_argument(
        "--train_csv",
        default=default_train_csv(),
        help="Training CSV with mode paths and good/bad labels (default: $NOVA_TRAIN_CSV or training_labels/tae_like_train.csv)",
    )
    ap.add_argument(
        "--data_dir",
        default=os.environ.get("NOVA_DATA"),
        help="Data directory used to resolve relative mode paths in --train_csv (default: $NOVA_DATA)",
    )
    ap.add_argument("--model_out", default="nova_cnn.pt", help="Output checkpoint path")
    ap.add_argument("--test_frac", type=float, default=0.2, help="Stratified test fraction")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split and training")
    ap.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    ap.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of epochs for split training and optional full-data refit",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=2e-2,
        help="Peak OneCycleLR learning rate for split training and full-data refit",
    )
    ap.add_argument(
        "--pos_weight",
        default=None,
        help=(
            "Positive-class weight for BCEWithLogitsLoss, where positive=good. "
            "Use 'auto' to compute n_bad/n_good from the training labels, a "
            "positive number to force a value, or omit/none for unweighted loss."
        ),
    )
    ap.add_argument(
        "--normalize",
        choices=["none", "standard", "robust", "maxabs"],
        default="robust",
        help="Per-mode image normalization",
    )
    ap.add_argument(
        "--eval_threshold",
        type=float,
        default=0.5,
        help="Probability threshold used for final metrics and saved checkpoint",
    )
    ap.add_argument("--M_target", type=int, default=54, help="Poloidal harmonics kept after m-axis pad/crop")
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
            "model on the full labeled CSV using the same OneCycle/clipping "
            "recipe for --epochs epochs, then save that full-data model. "
            "Held-out metrics still come from the best split-training checkpoint."
        ),
    )
    ap.add_argument(
        "--onecycle_div_factor",
        type=float,
        default=20.0,
        help=(
            "OneCycleLR initial divisor: initial_lr = --lr / value "
            "(default: 20, so --lr 0.02 starts at 0.001)"
        ),
    )
    ap.add_argument(
        "--onecycle_final_div_factor",
        type=float,
        default=100.0,
        help="OneCycleLR final divisor relative to initial_lr (default: 100)",
    )
    ap.add_argument(
        "--onecycle_pct_start",
        type=float,
        default=0.1,
        help="Fraction of training steps spent increasing LR to --lr (default: 0.1)",
    )
    ap.add_argument(
        "--grad_clip_norm",
        type=parse_optional_positive_float,
        default=1.0,
        help=(
            "Maximum gradient norm during split training and full refit (default: 1.0). "
            "Use none, off, or 0 to disable clipping."
        ),
    )
    return Config(**vars(ap.parse_args()))


def resolve_pos_weight(spec: str | None, items: list[dict[str, Any]], label: str) -> float | None:
    if spec is None or str(spec).strip().lower() in ("", "none", "off", "false"):
        return None

    value = str(spec).strip().lower()
    if value == "auto":
        n_good = sum(1 for it in items if it["label"] == 1)
        n_bad = sum(1 for it in items if it["label"] == 0)
        if n_good == 0 or n_bad == 0:
            raise ValueError(f"Cannot compute --pos_weight auto for {label}: need both good and bad labels.")
        return float(n_bad / n_good)

    try:
        weight = float(value)
    except ValueError as exc:
        raise ValueError("--pos_weight must be a positive number, 'auto', or 'none'") from exc

    if weight <= 0.0:
        raise ValueError("--pos_weight must be positive")
    return weight


def build_onecycle_training(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.OneCycleLR]:
    initial_lr = cfg.lr / cfg.onecycle_div_factor
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=len(loader),
        pct_start=cfg.onecycle_pct_start,
        anneal_strategy="cos",
        div_factor=cfg.onecycle_div_factor,
        final_div_factor=cfg.onecycle_final_div_factor,
        cycle_momentum=False,
    )
    return optimizer, scheduler


def describe_training_recipe(cfg: Config) -> str:
    initial_lr = cfg.lr / cfg.onecycle_div_factor
    final_lr = initial_lr / cfg.onecycle_final_div_factor
    grad_clip = "disabled" if cfg.grad_clip_norm is None else f"{cfg.grad_clip_norm:.3g}"
    return (
        f"OneCycleLR initial_lr={initial_lr:.3g}, max_lr={cfg.lr:.3g}, "
        f"pct_start={cfg.onecycle_pct_start:.3g}, final_lr={final_lr:.3g}, "
        f"grad_clip_norm={grad_clip}"
    )


def main():
    cfg = parse_args()

    if cfg.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if cfg.onecycle_div_factor <= 0.0:
        raise ValueError("--onecycle_div_factor must be positive")
    if cfg.onecycle_final_div_factor <= 0.0:
        raise ValueError("--onecycle_final_div_factor must be positive")
    if not 0.0 < cfg.onecycle_pct_start < 1.0:
        raise ValueError("--onecycle_pct_start must be between 0 and 1")
    seed_everything(cfg.seed)

    items = read_train_csv(cfg.train_csv, data_root=cfg.data_dir)
    train_items, test_items = train_test_split_stratified(items, cfg.test_frac, cfg.seed)
    train_pos_weight = resolve_pos_weight(cfg.pos_weight, train_items, "train split")

    print(f"Training CSV: {cfg.train_csv}")
    print(f"Data dir: {cfg.data_dir or '$NOVA_DATA'}")
    print(f"Total modes: {len(items)} | Train: {len(train_items)} | Test: {len(test_items)}")
    print(f"Raw preprocessing: R_target={cfg.R_target}, M_target={cfg.M_target}")
    print(f"Training recipe: {describe_training_recipe(cfg)}")
    if train_pos_weight is None:
        print("Loss pos_weight: none")
    else:
        print(f"Loss pos_weight: {train_pos_weight:.6g} (positive class = good)")

    train_ds = NovaModeDataset(
        train_items,
        normalize=cfg.normalize,
        M_target=cfg.M_target,
        R_target=cfg.R_target,
        cache_data=cfg.cache_data,
    )
    test_ds = NovaModeDataset(
        test_items,
        normalize=cfg.normalize,
        M_target=cfg.M_target,
        R_target=cfg.R_target,
        cache_data=cfg.cache_data,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    device = select_torch_device(cfg.device)
    print_torch_device_report(device)

    model = SmallCNN(in_ch=1).to(device)
    opt, sched = build_onecycle_training(model, train_loader, cfg)

    best_acc = -1.0
    best_epoch = 0
    best_state = None

    for ep in range(1, cfg.epochs + 1):
        epoch_t0 = time.perf_counter()
        loss = train_epoch(
            model,
            train_loader,
            opt,
            device,
            pos_weight=train_pos_weight,
            batch_scheduler=sched,
            grad_clip_norm=cfg.grad_clip_norm,
        )
        acc, probs, y_true, _ = eval_model(model, test_loader, device, thr=cfg.eval_threshold)

        if acc > best_acc:
            best_acc = acc
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        report_epoch = ep == 1 or ep == cfg.epochs or ep % 5 == 0
        if report_epoch:
            lr = opt.param_groups[0]["lr"]
            elapsed = time.perf_counter() - epoch_t0
            print(
                f"Epoch {ep:3d}/{cfg.epochs} | loss={loss:.4f} | "
                f"test_acc={acc:.4f} | lr={lr:.2e} | elapsed={elapsed:.1f}s",
                flush=True,
            )
            health = summarize_prediction_health(probs, y_true, cfg.eval_threshold)
            report_prediction_health("Split-test", ep, health)

    print(f"Best test acc: {best_acc:.4f} (epoch {best_epoch})")
    if best_state is not None:
        model.load_state_dict(best_state)

    acc, probs, y_true, paths = eval_model(model, test_loader, device, thr=cfg.eval_threshold)
    y_pred = (probs >= cfg.eval_threshold).astype(int)

    wrong = np.where(y_pred != y_true)[0]
    print("\nMisclassified modes:")
    print("Path,  true_label,  pred_label,  p_good")
    for i in wrong:
        true_lab = "good" if y_true[i] == 1 else "bad"
        pred_lab = "good" if y_pred[i] == 1 else "bad"
        print(f"{paths[i]}, {true_lab}, {pred_lab}, {probs[i]:.3f}")

    if np.any(y_true == 1):
        print("\nGood modes p_good range:", probs[y_true == 1].min(), probs[y_true == 1].max())
        print("mean p_good | true good:", probs[y_true == 1].mean())
    if np.any(y_true == 0):
        print("Bad modes p_good range:", probs[y_true == 0].min(), probs[y_true == 0].max())
        print("mean p_good | true bad :", probs[y_true == 0].mean())

    final_prediction_health = summarize_prediction_health(
        probs,
        y_true,
        cfg.eval_threshold,
    )

    print("\nConfusion matrix (rows=actual [bad,good], cols=pred [bad,good]):")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["bad", "good"]))

    preprocess_meta = build_raw_preprocess_metadata(
        R_target=cfg.R_target,
        M_target=cfg.M_target,
    )

    model_to_save = model
    saved_training_scope = "train_split"
    final_train_epochs = 0
    final_train_size = len(train_items)
    final_pos_weight = train_pos_weight

    if cfg.refit_full_before_save:
        if best_epoch <= 0:
            raise RuntimeError("Cannot refit on the full set because no best epoch was selected.")

        final_pos_weight = resolve_pos_weight(cfg.pos_weight, items, "full CSV refit")
        print(
            f"\nRefitting final raw CNN on all {len(items)} labeled modes "
            f"for {cfg.epochs} epochs before saving.",
            flush=True,
        )
        print(f"Full-fit recipe: {describe_training_recipe(cfg)}", flush=True)
        if final_pos_weight is not None:
            print(f"Full-fit loss pos_weight: {final_pos_weight:.6g} (positive class = good)")
        full_ds = NovaModeDataset(
            items,
            normalize=cfg.normalize,
            M_target=cfg.M_target,
            R_target=cfg.R_target,
            cache_data=cfg.cache_data,
        )
        seed_everything(cfg.seed)
        full_loader = DataLoader(full_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        full_eval_loader = DataLoader(
            full_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )
        final_model = SmallCNN(in_ch=1).to(device)
        final_opt, final_sched = build_onecycle_training(final_model, full_loader, cfg)

        for ep in range(1, cfg.epochs + 1):
            epoch_t0 = time.perf_counter()
            loss = train_epoch(
                final_model,
                full_loader,
                final_opt,
                device,
                pos_weight=final_pos_weight,
                batch_scheduler=final_sched,
                grad_clip_norm=cfg.grad_clip_norm,
            )
            report_epoch = ep == 1 or ep == cfg.epochs or ep % 5 == 0
            if report_epoch:
                lr = final_opt.param_groups[0]["lr"]
                _, full_probs, full_y_true, _ = eval_model(
                    final_model,
                    full_eval_loader,
                    device,
                    thr=cfg.eval_threshold,
                )
                final_prediction_health = summarize_prediction_health(
                    full_probs,
                    full_y_true,
                    cfg.eval_threshold,
                )
                elapsed = time.perf_counter() - epoch_t0
                print(
                    f"Full-fit epoch {ep:3d}/{cfg.epochs} | loss={loss:.4f} | "
                    f"lr={lr:.2e} | elapsed={elapsed:.1f}s",
                    flush=True,
                )
                report_prediction_health("Full-fit", ep, final_prediction_health)

        model_to_save = final_model
        saved_training_scope = "full_csv_refit"
        final_train_epochs = cfg.epochs
        final_train_size = len(items)

    torch.save(
        {
            "model_state_dict": model_to_save.state_dict(),
            "seed": cfg.seed,
            "normalize": cfg.normalize,
            "pos_weight_arg": cfg.pos_weight,
            "pos_weight": final_pos_weight,
            "initial_pos_weight": train_pos_weight,
            "final_pos_weight": final_pos_weight,
            "threshold": cfg.eval_threshold,
            "best_test_acc": best_acc,
            "best_epoch": best_epoch,
            "test_frac": cfg.test_frac,
            "initial_train_size": len(train_items),
            "test_size": len(test_items),
            "refit_full_before_save": cfg.refit_full_before_save,
            "saved_training_scope": saved_training_scope,
            "final_train_epochs": final_train_epochs,
            "final_train_size": final_train_size,
            "training_scheduler": "onecycle",
            "onecycle_max_lr": cfg.lr,
            "onecycle_div_factor": cfg.onecycle_div_factor,
            "onecycle_final_div_factor": cfg.onecycle_final_div_factor,
            "onecycle_pct_start": cfg.onecycle_pct_start,
            "grad_clip_norm": cfg.grad_clip_norm,
            "collapse_check_start_epoch": COLLAPSE_CHECK_START_EPOCH,
            "collapse_class_fraction": COLLAPSE_CLASS_FRACTION,
            "collapse_prob_std": COLLAPSE_PROB_STD,
            "final_prediction_health": (
                None
                if final_prediction_health is None
                else {
                    "n_samples": final_prediction_health.n_samples,
                    "n_true_good": final_prediction_health.n_true_good,
                    "n_predicted_good": final_prediction_health.n_predicted_good,
                    "predicted_good_fraction": final_prediction_health.predicted_good_fraction,
                    "true_good_fraction": final_prediction_health.true_good_fraction,
                    "prob_mean": final_prediction_health.prob_mean,
                    "prob_std": final_prediction_health.prob_std,
                    "prob_min": final_prediction_health.prob_min,
                    "prob_max": final_prediction_health.prob_max,
                    "collapse_detected": final_prediction_health.collapse_detected,
                    "collapse_reasons": list(final_prediction_health.collapse_reasons),
                }
            ),
            "model_type": "cnn_raw",
            "checkpoint_version": CHECKPOINT_VERSION,
            "preprocess": preprocess_meta,
            **preprocess_meta,
        },
        cfg.model_out,
    )
    print(f"\nSaved CNN to {cfg.model_out} ({saved_training_scope})")


if __name__ == "__main__":
    main()
