import numpy as np
import torch
import torch.nn as nn

# --- copy these from your training script ---
TARGET_M, TARGET_R = 54, 201

def load_mode_from_nova(path: str):
    f1 = np.fromfile(path)
    omega = f1[0]
    nr = int(f1[-3])
    gamma_d = f1[-2]
    ntor = f1[-1]
    nhar = int((f1.size-4)/(3*nr))
    f11 = f1[1:-3].reshape(3, nhar, nr)
    mode = f11[0, :, :]
    return mode, omega, gamma_d, ntor

def pad_or_crop(mode, Mt=TARGET_M, Rt=TARGET_R):
    mode = np.asarray(mode, dtype=np.float32)
    M, R = mode.shape
    out = np.zeros((Mt, Rt), dtype=np.float32)
    out[:min(M, Mt), :min(R, Rt)] = mode[:min(M, Mt), :min(R, Rt)]
    return out

def normalize_robust(x):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad < 1e-3:
        return x - med
    return (x - med) / (mad + 1e-8)

class SmallCNN(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x).squeeze(1)

def classify(path, model_path="nova_cnn.pt", threshold=0.55):
    ckpt = torch.load(model_path, map_location="cpu")
    model = SmallCNN(in_ch=1)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mode, omega, gamma_d, ntor = load_mode_from_nova(path)
    mode = pad_or_crop(mode)
    x = mode[None, None, :, :]  # (1,1,M,R)

    # use same normalization as training
    norm = ckpt.get("normalize", "robust")
    if norm == "robust":
        x = normalize_robust(x)
    # if norm == "none": do nothing

    x = torch.from_numpy(x.astype(np.float32))
    with torch.no_grad():
        logit = model(x)
        p_good = torch.sigmoid(logit).item()

    label = "good" if p_good >= threshold else "bad"
    print(f"{p}\n threshold used = {threshold}")
    return p_good, label, omega, gamma_d, ntor

if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    p_good, label, omega, gamma_d, ntor = classify(p)
    print(f"{p}\n  p_good={p_good:.3f}  label={label}\n  omega={omega:.4g}  gamma_d={gamma_d:.4g}  ntor={ntor:.0f}")



