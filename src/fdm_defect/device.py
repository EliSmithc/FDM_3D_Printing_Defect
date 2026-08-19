"""Runtime device selection.

Development happens on Apple Silicon and longer runs on an NVIDIA machine, so the
device is resolved at call time rather than baked into configs or checkpoints.
"""

from __future__ import annotations

import torch


def resolve_device(preference: str | None = None) -> torch.device:
    """Pick the best available device, or honour an explicit preference.

    Order is cuda, then mps, then cpu. An explicit preference is validated so a typo or
    an unavailable backend fails loudly instead of silently training on the cpu.
    """
    if preference:
        device = torch.device(preference)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"
