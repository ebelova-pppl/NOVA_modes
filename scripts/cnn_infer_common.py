from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from cont_features import load_datcon_for_mode, continuum_scalars
from mode_csv import read_mode_paths_csv as read_mode_paths_csv_shared
from mode_transform import resample_r, straighten_mode_window
from nova_mode_loader import load_mode_from_nova


LEGACY_PREPROCESS_DEFAULTS = {
    "R_target": 201,
    "M": 8,
    "center_power": 2.0,
    "median_k": 3,
    "max_step": 2,
}

RAW_PREPROCESS_DEFAULTS = {
    "R_target": 201,
    "M_target": 54,
}

CHECKPOINT_VERSION = 2
CONTINUUM_CHANNEL_CLIP_DEFAULT = 5.0
SUPPORTED_MODEL_KINDS = {
    "cnn_raw",
    "cnn_raw_continuum",
    "cnn_raw_continuum_branch",
    "cnn_straightened",
    "cnn_hybrid",
}
_WARNED_LEGACY_CHECKPOINTS: set[str] = set()


class UnsupportedCheckpointError(RuntimeError):
    pass


def normalize_mode_array(x: np.ndarray, normalize: str) -> np.ndarray:
    if normalize == "none":
        return x
    if normalize == "robust":
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        if mad < 1e-3:
            return x - med
        return (x - med) / (mad + 1e-8)
    if normalize == "maxabs":
        scale = float(np.max(np.abs(x))) + 1e-8
        return x / scale
    if normalize == "standard":
        mu = float(np.mean(x))
        sig = float(np.std(x)) + 1e-8
        return (x - mu) / sig
    raise ValueError("normalize must be none|standard|robust|maxabs")


def get_cont_scalars(mode_path: str, mode: np.ndarray, omega: float) -> dict[str, float]:
    """
    Match the hybrid training fallback exactly: missing or unreadable datcon
    silently falls back to safe defaults.
    """
    n_r = mode.shape[1]
    r = np.linspace(0.0, 1.0, n_r)

    try:
        low2, high2, *_ = load_datcon_for_mode(mode_path, n_r=n_r)
        cont = continuum_scalars(mode, omega, low2, high2, r=r)
        return {
            "delta2_eff": float(cont["delta2_eff"]),
            "r_star": float(cont["r_star"]),
            "S": float(cont["S"]),
            "W_star": float(cont["W_star"]),
            "has_cont": 1.0,
        }
    except Exception:
        return {
            "delta2_eff": 1e3,
            "r_star": 0.0,
            "S": 1e3,
            "W_star": 0.0,
            "has_cont": 0.0,
        }


def build_hybrid_scalar_vector(
    mode_path: str,
    mode: np.ndarray,
    omega: float,
    gamma_d: float,
    ntor: int,
) -> np.ndarray:
    cont = get_cont_scalars(mode_path, mode, omega)
    return np.array(
        [
            float(gamma_d),
            float(cont["delta2_eff"]),
            float(cont["W_star"]),
            float(cont["S"]),
            float(cont["r_star"]),
            float(omega),
            float(ntor),
            float(cont["has_cont"]),
        ],
        dtype=np.float32,
    )


def build_preprocess_metadata(
    *,
    R_target: int,
    M: int,
    center_power: float,
    median_k: int,
    max_step: int,
) -> dict[str, Any]:
    return {
        "R_target": int(R_target),
        "M": int(M),
        "center_power": float(center_power),
        "median_k": int(median_k),
        "max_step": int(max_step),
    }


def build_raw_preprocess_metadata(
    *,
    R_target: int,
    M_target: int,
    continuum_channels: bool = False,
    continuum_branch: bool = False,
    continuum_clip: float = CONTINUUM_CHANNEL_CLIP_DEFAULT,
) -> dict[str, Any]:
    return {
        "R_target": int(R_target),
        "M_target": int(M_target),
        "continuum_channels": bool(continuum_channels),
        "continuum_branch": bool(continuum_branch),
        "continuum_clip": float(continuum_clip),
    }


def resolve_preprocess_metadata(
    checkpoint: Mapping[str, Any],
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    preprocess_block = checkpoint.get("preprocess")
    if not isinstance(preprocess_block, Mapping):
        preprocess_block = {}

    resolved: dict[str, Any] = {}
    missing: list[str] = []

    for key, default in LEGACY_PREPROCESS_DEFAULTS.items():
        value = preprocess_block.get(key)
        if value is None:
            value = checkpoint.get(key)
        if value is None:
            value = default
            missing.append(key)
        resolved[key] = value

    resolved["R_target"] = int(resolved["R_target"])
    resolved["M"] = int(resolved["M"])
    resolved["center_power"] = float(resolved["center_power"])
    resolved["median_k"] = int(resolved["median_k"])
    resolved["max_step"] = int(resolved["max_step"])

    if missing:
        warning_key = f"{checkpoint_path or '<checkpoint>'}:{','.join(missing)}"
        if warning_key not in _WARNED_LEGACY_CHECKPOINTS:
            warnings.warn(
                (
                    f"{checkpoint_path or 'checkpoint'} does not contain preprocessing "
                    f"metadata for {', '.join(missing)}; using legacy defaults "
                    f"{LEGACY_PREPROCESS_DEFAULTS}"
                ),
                category=UserWarning,
                stacklevel=2,
            )
            _WARNED_LEGACY_CHECKPOINTS.add(warning_key)

    return resolved


def resolve_raw_preprocess_metadata(
    checkpoint: Mapping[str, Any],
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    preprocess_block = checkpoint.get("preprocess")
    if not isinstance(preprocess_block, Mapping):
        preprocess_block = {}

    resolved: dict[str, Any] = {}
    missing: list[str] = []

    for key, default in RAW_PREPROCESS_DEFAULTS.items():
        value = preprocess_block.get(key)
        if value is None:
            value = checkpoint.get(key)
        if value is None:
            value = default
            missing.append(key)
        resolved[key] = value

    resolved["R_target"] = int(resolved["R_target"])
    resolved["M_target"] = int(resolved["M_target"])
    continuum_channels = preprocess_block.get("continuum_channels")
    if continuum_channels is None:
        continuum_channels = checkpoint.get("continuum_channels", False)
    continuum_branch = preprocess_block.get("continuum_branch")
    if continuum_branch is None:
        continuum_branch = checkpoint.get("continuum_branch", False)
    continuum_clip = preprocess_block.get("continuum_clip")
    if continuum_clip is None:
        continuum_clip = checkpoint.get(
            "continuum_clip",
            CONTINUUM_CHANNEL_CLIP_DEFAULT,
        )
    resolved["continuum_channels"] = bool(continuum_channels)
    resolved["continuum_branch"] = bool(continuum_branch)
    resolved["continuum_clip"] = float(continuum_clip)

    if missing:
        warning_key = f"{checkpoint_path or '<checkpoint>'}:raw:{','.join(missing)}"
        if warning_key not in _WARNED_LEGACY_CHECKPOINTS:
            warnings.warn(
                (
                    f"{checkpoint_path or 'checkpoint'} does not contain raw preprocessing "
                    f"metadata for {', '.join(missing)}; using legacy defaults "
                    f"{RAW_PREPROCESS_DEFAULTS}"
                ),
                category=UserWarning,
                stacklevel=2,
            )
            _WARNED_LEGACY_CHECKPOINTS.add(warning_key)

    return resolved


def pad_or_crop_raw(mode: np.ndarray, M_target: int = 54, R_target: int = 201) -> np.ndarray:
    mode = np.asarray(mode, dtype=np.float32)
    n_m, n_r = mode.shape
    out = np.zeros((M_target, R_target), dtype=np.float32)
    mmin = min(n_m, M_target)
    rmin = min(n_r, R_target)
    out[:mmin, :rmin] = mode[:mmin, :rmin]
    return out


def _interp_finite_1d(values: np.ndarray, R_target: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    out = np.zeros(R_target, dtype=np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return out
    if np.count_nonzero(valid) == 1:
        out.fill(float(arr[valid][0]))
        return out

    r_src = np.linspace(0.0, 1.0, arr.size, dtype=np.float32)
    r_tgt = np.linspace(0.0, 1.0, R_target, dtype=np.float32)
    return np.interp(r_tgt, r_src[valid], arr[valid]).astype(np.float32)


def build_continuum_channel_array(
    mode_path: str,
    omega: float,
    *,
    n_r: int,
    M_target: int,
    R_target: int,
    clip: float = CONTINUUM_CHANNEL_CLIP_DEFAULT,
) -> np.ndarray:
    """
    Build broadcasted ``du`` and ``dl`` continuum channels for raw-CNN input.

    ``du = (sqrt(high2) - omega) / omega`` and
    ``dl = (omega - sqrt(low2)) / omega``. Positive values for both channels
    mean the mode frequency lies inside the local gap. Missing continuum data
    falls back to zero-valued channels for this first two-channel experiment.
    """
    omega_value = float(omega)
    if not np.isfinite(omega_value) or omega_value == 0.0:
        raise ValueError(f"omega must be finite and nonzero, got {omega}")
    if not np.isfinite(clip) or float(clip) <= 0.0:
        raise ValueError(f"continuum clip must be positive and finite, got {clip}")

    try:
        low2, high2, *_ = load_datcon_for_mode(mode_path, n_r=n_r)
    except Exception:
        return np.zeros((2, M_target, R_target), dtype=np.float32)

    valid = (
        np.isfinite(low2)
        & np.isfinite(high2)
        & (low2 >= 0.0)
        & (high2 >= 0.0)
        & (high2 >= low2)
    )
    if not np.any(valid):
        return np.zeros((2, M_target, R_target), dtype=np.float32)

    low = np.full(n_r, np.nan, dtype=np.float32)
    high = np.full(n_r, np.nan, dtype=np.float32)
    low[valid] = np.sqrt(low2[valid]).astype(np.float32)
    high[valid] = np.sqrt(high2[valid]).astype(np.float32)
    du = (high - omega_value) / omega_value
    dl = (omega_value - low) / omega_value

    du_rs = np.clip(_interp_finite_1d(du, R_target), -clip, clip)
    dl_rs = np.clip(_interp_finite_1d(dl, R_target), -clip, clip)
    channels = np.stack([du_rs, dl_rs]).astype(np.float32)
    return np.broadcast_to(channels[:, None, :], (2, M_target, R_target)).copy()


def build_continuum_branch_array(
    mode_path: str,
    mode: np.ndarray,
    omega: float,
    *,
    R_target: int,
    clip: float = CONTINUUM_CHANNEL_CLIP_DEFAULT,
) -> np.ndarray:
    """Build ``W_norm``, ``du``, ``dl``, and validity-mask radial channels."""
    mode = np.asarray(mode)
    if mode.ndim != 2 or mode.shape[1] < 1:
        raise ValueError(f"mode must be a nonempty 2D array, got shape {mode.shape}")

    omega_value = float(omega)
    if not np.isfinite(omega_value) or omega_value <= 0.0:
        raise ValueError(f"omega must be positive and finite, got {omega}")
    if not np.isfinite(clip) or float(clip) <= 0.0:
        raise ValueError(f"continuum clip must be positive and finite, got {clip}")

    mode_rs = resample_r(mode, R_target=R_target)
    radial_energy = np.sum(np.abs(mode_rs) ** 2, axis=0)
    peak_energy = float(np.max(radial_energy))
    if peak_energy > 0.0:
        w_norm = radial_energy / peak_energy
    else:
        w_norm = np.zeros(R_target, dtype=np.float32)

    branch = np.zeros((4, R_target), dtype=np.float32)
    branch[0] = np.asarray(w_norm, dtype=np.float32)

    try:
        low2, high2, *_ = load_datcon_for_mode(mode_path, n_r=mode.shape[1])
    except Exception:
        return branch

    valid = (
        np.isfinite(low2)
        & np.isfinite(high2)
        & (low2 >= 0.0)
        & (high2 >= 0.0)
        & (high2 >= low2)
    )
    if not np.any(valid):
        return branch

    low = np.full(mode.shape[1], np.nan, dtype=np.float32)
    high = np.full(mode.shape[1], np.nan, dtype=np.float32)
    low[valid] = np.sqrt(low2[valid]).astype(np.float32)
    high[valid] = np.sqrt(high2[valid]).astype(np.float32)
    du = (high - omega_value) / omega_value
    dl = (omega_value - low) / omega_value

    nearest_src = np.rint(
        np.linspace(0.0, valid.size - 1, R_target, dtype=np.float32)
    ).astype(int)
    valid_rs = valid[nearest_src]
    branch[1] = np.clip(_interp_finite_1d(du, R_target), -clip, clip)
    branch[2] = np.clip(_interp_finite_1d(dl, R_target), -clip, clip)
    branch[1, ~valid_rs] = 0.0
    branch[2, ~valid_rs] = 0.0
    branch[3] = valid_rs.astype(np.float32)
    return branch


def build_raw_image_array(
    mode_path: str,
    mode: np.ndarray,
    omega: float,
    *,
    normalize: str,
    M_target: int,
    R_target: int,
    continuum_channels: bool = False,
    continuum_clip: float = CONTINUUM_CHANNEL_CLIP_DEFAULT,
) -> np.ndarray:
    mode_rs = resample_r(mode, R_target=R_target)
    x_img = pad_or_crop_raw(
        mode_rs,
        M_target=M_target,
        R_target=R_target,
    )
    x_img = x_img[None, :, :]
    x_img = normalize_mode_array(x_img, normalize)
    x_img = np.asarray(x_img, dtype=np.float32)

    if not continuum_channels:
        return x_img

    cont_channels = build_continuum_channel_array(
        mode_path,
        omega,
        n_r=mode.shape[1],
        M_target=M_target,
        R_target=R_target,
        clip=continuum_clip,
    )
    return np.concatenate([x_img, cont_channels], axis=0).astype(np.float32)


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int = 1):
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
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x).squeeze(1)
        return x


class ContinuumBranchCNN(nn.Module):
    """Raw mode-image CNN with radius-aligned continuum feature fusion."""

    def __init__(self):
        super().__init__()
        self.mode_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.continuum_features = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(64 + 16, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x_img: torch.Tensor,
        x_continuum: torch.Tensor,
    ) -> torch.Tensor:
        z_mode = self.mode_features(x_img).mean(dim=2)
        z_continuum = self.continuum_features(x_continuum)
        if z_mode.shape[-1] != z_continuum.shape[-1]:
            raise ValueError(
                "Mode and continuum radial feature grids do not align: "
                f"{z_mode.shape[-1]} != {z_continuum.shape[-1]}"
            )
        z = self.fusion(torch.cat([z_mode, z_continuum], dim=1))
        return self.head(self.pool(z)).squeeze(1)


class HybridCNN(nn.Module):
    def __init__(self, n_scalars: int = 8, in_ch: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.sc_fc = nn.Sequential(
            nn.Linear(n_scalars, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x_img: torch.Tensor, x_sc: torch.Tensor) -> torch.Tensor:
        z_img = self.features(x_img)
        z_img = self.pool(z_img)
        z_img = self.img_fc(z_img)
        z_sc = self.sc_fc(x_sc)
        z = torch.cat([z_img, z_sc], dim=1)
        return self.head(z).squeeze(1)


def _load_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    # Local checkpoints in this repo may contain numpy arrays, so PyTorch 2.6+
    # needs weights_only=False to load the full checkpoint payload.
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise UnsupportedCheckpointError(
            f"{checkpoint_path} is not a supported CNN checkpoint payload."
        )

    if "model_state_dict" in checkpoint:
        return checkpoint

    state_dict = checkpoint.get("state_dict")
    if isinstance(state_dict, Mapping):
        normalized = dict(checkpoint)
        normalized["model_state_dict"] = state_dict
        return normalized

    if _looks_like_model_state_dict(checkpoint):
        return {"model_state_dict": checkpoint}

    raise UnsupportedCheckpointError(
        f"{checkpoint_path} is not a supported CNN checkpoint payload."
    )


def _looks_like_model_state_dict(payload: Mapping[str, Any]) -> bool:
    if not payload:
        return False
    prefixes = {str(key).split(".")[0] for key in payload}
    if not (
        prefixes
        & {
            "features",
            "mode_features",
            "continuum_features",
            "fusion",
            "head",
            "img_fc",
            "sc_fc",
            "module",
        }
    ):
        return False
    return all(torch.is_tensor(value) for value in payload.values())


def infer_checkpoint_kind(
    checkpoint: Mapping[str, Any],
    model_kind: str = "auto",
    checkpoint_path: str | None = None,
) -> str:
    if model_kind != "auto":
        if model_kind not in SUPPORTED_MODEL_KINDS:
            raise UnsupportedCheckpointError(
                f"Unsupported model kind '{model_kind}'. "
                f"Expected one of: auto, {', '.join(sorted(SUPPORTED_MODEL_KINDS))}."
            )
        return model_kind

    state_dict = checkpoint.get("model_state_dict", {})
    prefixes = {key.split(".")[0] for key in state_dict}

    if (
        "scalars_mu" in checkpoint
        or "scalars_sig" in checkpoint
        or "img_fc" in prefixes
        or "sc_fc" in prefixes
    ):
        return "cnn_hybrid"
    if {"mode_features", "continuum_features", "fusion"}.issubset(prefixes):
        return "cnn_raw_continuum_branch"

    model_type = checkpoint.get("model_type")
    if model_type in {"cnn_hybrid", "hybrid", "hybrid_cnn"}:
        return "cnn_hybrid"
    if model_type in {
        "cnn_raw_continuum_branch",
        "raw_continuum_branch",
    }:
        return "cnn_raw_continuum_branch"
    if model_type in {"cnn_raw_continuum", "raw_continuum"}:
        return "cnn_raw_continuum"
    if model_type in {"cnn_straightened", "straightened"}:
        return "cnn_straightened"
    if model_type in {"cnn_raw", "raw"}:
        return "cnn_raw"

    if "preprocess" in checkpoint:
        preprocess = checkpoint.get("preprocess")
        if isinstance(preprocess, Mapping) and "M_target" in preprocess:
            if bool(preprocess.get("continuum_branch", False)):
                return "cnn_raw_continuum_branch"
            if bool(preprocess.get("continuum_channels", False)):
                return "cnn_raw_continuum"
            return "cnn_raw"
        return "cnn_straightened"
    if "M_target" in checkpoint:
        return "cnn_raw"
    if any(key in checkpoint for key in LEGACY_PREPROCESS_DEFAULTS):
        return "cnn_straightened"

    if checkpoint_path:
        name = Path(checkpoint_path).name.lower()
        if "hybrid" in name:
            return "cnn_hybrid"
        if "continuum_branch" in name and "raw" in name:
            return "cnn_raw_continuum_branch"
        if "continuum" in name and "raw" in name:
            return "cnn_raw_continuum"
        if "straight" in name:
            return "cnn_straightened"
        if "raw" in name:
            return "cnn_raw"

    #if checkpoint.get("normalize") == "maxabs": ChatGPT did not like this heuristic since normalize could be maxabs for raw model type, but in practice all known straightened checkpoints have maxabs and no known hybrid checkpoints have maxabs, so this is actually a pretty strong signal. Leaving it out for now since it's a bit hacky, but it could be added back as a final heuristic if we find more ambiguous checkpoints in the future.
    #    return "cnn_straightened"

    raise UnsupportedCheckpointError(
        "Checkpoint does not look like a supported CNN checkpoint. "
        "If this is an older checkpoint, pass an explicit --model_kind."
    )


def _infer_hybrid_scalar_dim(checkpoint: Mapping[str, Any]) -> int:
    scalars_mu = checkpoint.get("scalars_mu")
    if scalars_mu is not None:
        return int(np.asarray(scalars_mu).shape[0])

    weight = checkpoint["model_state_dict"].get("sc_fc.0.weight")
    if weight is None:
        raise UnsupportedCheckpointError(
            "Hybrid checkpoint is missing scalar normalization arrays and scalar-layer weights."
        )
    return int(weight.shape[1])


@dataclass
class LoadedCNNClassifier:
    checkpoint_path: str
    checkpoint_kind: str
    model: nn.Module
    device: torch.device
    normalize: str
    threshold: float
    preprocess: dict[str, Any]
    scalars_mu: np.ndarray | None = None
    scalars_sig: np.ndarray | None = None

    def _prepare_image_tensor(
        self,
        mode_path: str,
        mode: np.ndarray,
        omega: float,
    ) -> torch.Tensor:
        if self.checkpoint_kind in {
            "cnn_raw",
            "cnn_raw_continuum",
            "cnn_raw_continuum_branch",
        }:
            x_img = build_raw_image_array(
                mode_path,
                mode,
                omega,
                normalize=self.normalize,
                M_target=self.preprocess["M_target"],
                R_target=self.preprocess["R_target"],
                continuum_channels=bool(self.preprocess.get("continuum_channels", False)),
                continuum_clip=float(
                    self.preprocess.get(
                        "continuum_clip",
                        CONTINUUM_CHANNEL_CLIP_DEFAULT,
                    )
                ),
            )
        else:
            mode_rs = resample_r(mode, R_target=self.preprocess["R_target"])
            x_img, _mc, _mc_int = straighten_mode_window(
                mode_rs,
                M=self.preprocess["M"],
                center_power=self.preprocess["center_power"],
                median_k=self.preprocess["median_k"],
                max_step=self.preprocess["max_step"],
            )
            x_img = x_img[None, :, :]
            x_img = normalize_mode_array(x_img, self.normalize)
        return torch.from_numpy(np.asarray(x_img, dtype=np.float32)).unsqueeze(0).to(self.device)

    def predict(
        self,
        mode_path: str,
        threshold: float | None = None,
        *,
        return_mode: bool = False,
    ) -> dict[str, Any]:
        resolved_mode_path = str(Path(mode_path).expanduser())
        mode, omega, gamma_d, ntor = load_mode_from_nova(resolved_mode_path)
        x_img = self._prepare_image_tensor(resolved_mode_path, mode, omega)

        with torch.no_grad():
            if self.checkpoint_kind == "cnn_raw_continuum_branch":
                x_continuum = build_continuum_branch_array(
                    resolved_mode_path,
                    mode,
                    omega,
                    R_target=self.preprocess["R_target"],
                    clip=float(
                        self.preprocess.get(
                            "continuum_clip",
                            CONTINUUM_CHANNEL_CLIP_DEFAULT,
                        )
                    ),
                )
                x_continuum_t = (
                    torch.from_numpy(x_continuum).unsqueeze(0).to(self.device)
                )
                logits = self.model(x_img, x_continuum_t)
            elif self.checkpoint_kind == "cnn_hybrid":
                mode_rs = resample_r(mode, R_target=self.preprocess["R_target"])
                x_sc = build_hybrid_scalar_vector(
                    resolved_mode_path,
                    mode_rs,
                    omega,
                    gamma_d,
                    ntor,
                )
                if self.scalars_mu is None or self.scalars_sig is None:
                    raise UnsupportedCheckpointError(
                        f"{self.checkpoint_path} is missing scalars_mu/scalars_sig needed for hybrid inference."
                    )
                x_sc = (x_sc - self.scalars_mu) / self.scalars_sig
                x_sc_t = torch.from_numpy(np.asarray(x_sc, dtype=np.float32)).unsqueeze(0).to(self.device)
                logits = self.model(x_img, x_sc_t)
            else:
                logits = self.model(x_img)

            p_good = float(torch.sigmoid(logits).item())

        used_threshold = self.threshold if threshold is None else float(threshold)
        result = {
            "path": resolved_mode_path,
            "label": "good" if p_good >= used_threshold else "bad",
            "p_good": p_good,
            "threshold": used_threshold,
            "omega": float(omega),
            "gamma_d": float(gamma_d),
            "ntor": int(ntor),
            "checkpoint_kind": self.checkpoint_kind,
        }
        if return_mode:
            result["mode"] = mode
        return result


def load_cnn_classifier(
    checkpoint_path: str,
    *,
    device: str | None = None,
    model_kind: str = "auto",
) -> LoadedCNNClassifier:
    resolved_checkpoint_path = str(Path(checkpoint_path).expanduser())
    checkpoint = _load_checkpoint(resolved_checkpoint_path)
    checkpoint_kind = infer_checkpoint_kind(
        checkpoint,
        model_kind=model_kind,
        checkpoint_path=resolved_checkpoint_path,
    )
    if checkpoint_kind in {
        "cnn_raw",
        "cnn_raw_continuum",
        "cnn_raw_continuum_branch",
    }:
        preprocess = resolve_raw_preprocess_metadata(
            checkpoint,
            checkpoint_path=resolved_checkpoint_path,
        )
    else:
        preprocess = resolve_preprocess_metadata(
            checkpoint,
            checkpoint_path=resolved_checkpoint_path,
        )

    torch_device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    if checkpoint_kind == "cnn_hybrid":
        n_scalars = _infer_hybrid_scalar_dim(checkpoint)
        model: nn.Module = HybridCNN(n_scalars=n_scalars, in_ch=1)
        scalars_mu_raw = checkpoint.get("scalars_mu")
        scalars_sig_raw = checkpoint.get("scalars_sig")
        scalars_mu = None if scalars_mu_raw is None else np.asarray(scalars_mu_raw, dtype=np.float32)
        scalars_sig = None if scalars_sig_raw is None else np.asarray(scalars_sig_raw, dtype=np.float32)
    elif checkpoint_kind == "cnn_raw_continuum_branch":
        model = ContinuumBranchCNN()
        scalars_mu = None
        scalars_sig = None
    else:
        in_ch = 3 if checkpoint_kind == "cnn_raw_continuum" else 1
        model = SmallCNN(in_ch=in_ch)
        scalars_mu = None
        scalars_sig = None

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise UnsupportedCheckpointError(
            f"{resolved_checkpoint_path} could not be loaded as {checkpoint_kind}: {exc}"
        ) from exc

    model.to(torch_device)
    model.eval()

    return LoadedCNNClassifier(
        checkpoint_path=resolved_checkpoint_path,
        checkpoint_kind=checkpoint_kind,
        model=model,
        device=torch_device,
        normalize=str(checkpoint.get("normalize", "maxabs")),
        threshold=float(checkpoint.get("threshold", 0.5)),
        preprocess=preprocess,
        scalars_mu=scalars_mu,
        scalars_sig=scalars_sig,
    )


def classify_mode_cnn(
    mode_path: str,
    checkpoint_path: str,
    device: str | None = None,
    threshold: float | None = None,
    *,
    model_kind: str = "auto",
) -> tuple[str, float]:
    classifier = load_cnn_classifier(
        checkpoint_path,
        device=device,
        model_kind=model_kind,
    )
    result = classifier.predict(mode_path, threshold=threshold)
    return result["label"], result["p_good"]


def classify_mode_cnn_full(
    mode_path: str,
    checkpoint_path: str,
    device: str | None = None,
    threshold: float | None = None,
    *,
    model_kind: str = "auto",
    return_mode: bool = False,
) -> dict[str, Any]:
    classifier = load_cnn_classifier(
        checkpoint_path,
        device=device,
        model_kind=model_kind,
    )
    return classifier.predict(mode_path, threshold=threshold, return_mode=return_mode)


def read_mode_paths_csv(csv_path: str, *, data_root: str | Path | None = None) -> list[str]:
    return read_mode_paths_csv_shared(csv_path, data_root=data_root)
