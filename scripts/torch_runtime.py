from __future__ import annotations

import os
import time

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
    print("Device:", device, flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"), flush=True)

    if device.type != "cuda":
        return

    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    print(f"CUDA device {idx}: {props.name}", flush=True)

    try:
        with torch.cuda.device(idx):
            free, total = torch.cuda.mem_get_info()
        print(f"CUDA memory free/total: {_format_bytes(free)} / {_format_bytes(total)}", flush=True)
    except RuntimeError as exc:
        print(f"CUDA memory free/total: unavailable ({exc})", flush=True)

    allocated = torch.cuda.memory_allocated(idx)
    reserved = torch.cuda.memory_reserved(idx)
    print(
        "CUDA memory allocated/reserved by this process: "
        f"{_format_bytes(allocated)} / {_format_bytes(reserved)}",
        flush=True,
    )


def run_smoke_test(requested: str | None = None) -> None:
    t0 = time.perf_counter()
    device = select_torch_device(requested)
    print(f"Timing: select_device={time.perf_counter() - t0:.3f}s", flush=True)

    t_report = time.perf_counter()
    print_torch_device_report(device)
    print(f"Timing: device_report={time.perf_counter() - t_report:.3f}s", flush=True)

    print("Running Torch smoke test...", flush=True)
    t_alloc = time.perf_counter()
    x = torch.ones((256, 256), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"Timing: allocate_tensor={time.perf_counter() - t_alloc:.3f}s", flush=True)

    t_matmul = time.perf_counter()
    y = (x @ x).mean()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"Timing: matmul={time.perf_counter() - t_matmul:.3f}s", flush=True)

    t_cpu = time.perf_counter()
    result = float(y.detach().cpu())
    print(f"Timing: copy_result_to_cpu={time.perf_counter() - t_cpu:.3f}s", flush=True)

    print(f"Smoke test OK: mean={result:.3f}", flush=True)
    if device.type == "cuda":
        print_torch_device_report(device)
    print(f"Timing: total={time.perf_counter() - t0:.3f}s", flush=True)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Report Torch CUDA device state.")
    ap.add_argument("--device", default=os.environ.get("NOVA_TORCH_DEVICE"))
    ap.add_argument("--smoke", action="store_true", help="Run a small tensor allocation/matmul test")
    args = ap.parse_args()

    if args.smoke:
        run_smoke_test(args.device)
    else:
        print_torch_device_report(select_torch_device(args.device))


if __name__ == "__main__":
    main()
