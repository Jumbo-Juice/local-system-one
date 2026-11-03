"""Which torch device the fraud app's own models use, and the few calls that differ per device.

The laptop is an Intel XPU and the NUC an NVIDIA GPU (fraud/PROGRESS.md, 2026-11-03). XPU stays first, so
the laptop behaves as before. Implementation choice.
"""

from __future__ import annotations

import torch


def _xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def default_device() -> str:
    if _xpu():
        return "xpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def synchronizer(dev: str):
    """What waits for the device to finish, so a step's time means something."""
    if dev.startswith("xpu"):
        return torch.xpu.synchronize
    if dev.startswith("cuda"):
        return torch.cuda.synchronize
    return lambda: None


def reset_peak_memory(dev: str) -> None:
    if dev.startswith("xpu"):
        torch.xpu.reset_peak_memory_stats()
    elif dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gib(dev: str) -> float | None:
    if dev.startswith("xpu"):
        return torch.xpu.max_memory_allocated() / 2**30
    if dev.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 2**30
    return None


def empty_cache() -> None:
    if _xpu():
        torch.xpu.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
