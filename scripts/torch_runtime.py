from __future__ import annotations

import os

import torch


def select_torch_device(requested: str | None = None) -> torch.device:
    """Choose a torch device, allowing an env override for batch scripts."""
    requested = requested or os.environ.get("NOVA_TORCH_DEVICE")
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested torch device '{requested}', but torch reports CUDA is not available."
            )
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _format_bytes(nbytes: int) -> str:
    gib = nbytes / 1024**3
    return f"{gib:.2f} GiB"


def print_torch_device_report(device: torch.device) -> None:
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

    if device.type != "cuda":
        return

    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    print(f"CUDA device {idx}: {props.name}")

    try:
        with torch.cuda.device(idx):
            free, total = torch.cuda.mem_get_info()
        print(f"CUDA memory free/total: {_format_bytes(free)} / {_format_bytes(total)}")
    except RuntimeError as exc:
        print(f"CUDA memory free/total: unavailable ({exc})")

    allocated = torch.cuda.memory_allocated(idx)
    reserved = torch.cuda.memory_reserved(idx)
    print(
        "CUDA memory allocated/reserved by this process: "
        f"{_format_bytes(allocated)} / {_format_bytes(reserved)}"
    )
