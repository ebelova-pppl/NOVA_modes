import os
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from nova_mode_loader import load_mode_from_nova
from path_utils import resolve_mode_csv_path
from paths import NOVA_TRAIN_CSV

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
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            p = resolve_mode_csv_path(row[0])
            lab = row[1].strip().lower()
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


# =========================
# 2) Dataset
# =========================
TARGET_M, TARGET_R = 54, 201

def pad_or_crop(mode, Mt=TARGET_M, Rt=TARGET_R):
    mode = np.asarray(mode, dtype=np.float32)
    M, R = mode.shape
    out = np.zeros((Mt, Rt), dtype=np.float32)
    mmin = min(M, Mt)
    rmin = min(R, Rt)
    out[:mmin, :rmin] = mode[:mmin, :rmin]
    return out

class NovaModeDataset(Dataset):
    def __init__(self, items, normalize="robust"):
        self.items = items
        self.normalize = normalize

    def _normalize(self, x):
        if self.normalize == "none":
            return x
        if self.normalize == "robust":
            med = float(np.median(x))
            mad = float(np.median(np.abs(x - med)))
            if mad < 1e-3:
                return x - med
            return (x - med) / (mad + 1e-8)
        if self.normalize == "standard":
            mu = float(np.mean(x))
            sig = float(np.std(x)) + 1e-8
            return (x - mu) / sig
        raise ValueError("normalize must be none|standard|robust")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        mode, omega, gamma_d, ntor = load_mode_from_nova(it["path"])

        mode = pad_or_crop(mode)           # (54,201)
        x = mode[None, :, :]               # (1,54,201)
        x = self._normalize(x)

        y = torch.tensor(it["label"], dtype=torch.long)
        return torch.from_numpy(x), y, it["path"]


# =========================
# 3) Collate function: pad variable (M,R) per batch
# =========================
# removed

# =========================
# 4) Small CNN: size-agnostic via AdaptiveAvgPool2d
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
            nn.Dropout(p=0.2),
            nn.Linear(32, 1)  # logits
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x).squeeze(1)
        return x


# =========================
# 5) Train / Eval loops
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
# 6) Main
# =========================
@dataclass
class Config:
    train_csv: str = str(NOVA_TRAIN_CSV)
    test_frac: float = 0.2
    seed: int = 42
    batch_size: int = 32
    epochs: int = 80
    lr: float = 5e-3
    normalize: str = "robust"  # "none" is OK too since max=1
    model_out: str = "nova_cnn.pt"


def main():
    cfg = Config()

    items = read_train_csv(cfg.train_csv)
    train_items, test_items = train_test_split_stratified(items, cfg.test_frac, cfg.seed)

    print(f"Total modes: {len(items)} | Train: {len(train_items)} | Test: {len(test_items)}")

    train_ds = NovaModeDataset(train_items, normalize=cfg.normalize)
    test_ds  = NovaModeDataset(test_items, normalize=cfg.normalize)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print("Device:", device)

    model = SmallCNN(in_ch=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_state = None

    for ep in range(1, cfg.epochs + 1):
        loss = train_epoch(model, train_loader, opt, device)
        acc, probs, y_true, _ = eval_model(model, test_loader, device)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 5 == 0:
            print(f"Epoch {ep:3d}/{cfg.epochs} | loss={loss:.4f} | test_acc={acc:.4f}")

    print(f"Best test acc: {best_acc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics
    acc, probs, y_true, paths = eval_model(model, test_loader, device)
    thr = 0.55
    y_pred = (probs >= thr).astype(int)
    #for thr in np.linspace(0.4, 0.75, 8):     # sweep thresholds to check nmber of FPs
        #y_pred = (probs >= thr).astype(int)
        #tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        #print(f"thr={thr:.2f}  FP={fp}  FN={fn}  TP={tp}")

    # Check FP cases
    wrong = np.where(y_pred != y_true)[0]
    print("\nMisclassified modes:")
    print("\nPath,  true_label,  pred_label,  p_good")
    for i in wrong:
        true_lab = "good" if y_true[i] == 1 else "bad"
        pred_lab = "good" if y_pred[i] == 1 else "bad"
        print(f"{paths[i]}, {true_lab}, {pred_lab}, {probs[i]:.3f}")

    print("Good modes p_good range:",
          probs[y_true==1].min(), probs[y_true==1].max())
    print("Bad modes p_good range:",
          probs[y_true==0].min(), probs[y_true==0].max())

    print("\nConfusion matrix (rows=actual [bad,good], cols=pred [bad,good]):")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["bad", "good"]))

    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "normalize": cfg.normalize,
        "threshold": 0.5,
    }, cfg.model_out)
    print(f"\nSaved CNN to {cfg.model_out}")


if __name__ == "__main__":
    main()
