"""Extracting frozen backbone features.

The first model of this project is a linear probe: run a pretrained backbone over every
frame once, cache the resulting vectors, then fit a linear classifier on top. It costs
one forward pass over the dataset and no training loop, and it answers the question
worth asking before building anything larger - how hard is this problem actually?

Caching the vectors means later experiments (different classifiers, different folds,
per-session aggregation) are seconds rather than minutes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

#: Self-supervised ViT. Its features transfer well without fine-tuning, which is what a
#: probe needs, and it was never trained on labels that might overlap this task.
DEFAULT_MODEL = "vit_small_patch14_dinov2.lvd142m"


class ImageListDataset(Dataset):
    """Loads a fixed list of image paths and applies a timm transform."""

    def __init__(self, paths: Sequence[Path], transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def load_backbone(model_name: str = DEFAULT_MODEL, device: torch.device | None = None):
    """Load a pretrained backbone with its classifier head removed."""
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    if device is not None:
        model.to(device)
    transform = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
    return model, transform


@torch.inference_mode()
def extract_embeddings(
    paths: Sequence[Path],
    model,
    transform,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
    progress: bool = True,
) -> np.ndarray:
    """Return an ``(n_images, n_features)`` float32 array in the order of ``paths``."""
    loader = DataLoader(
        ImageListDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,  # order must match `paths` so rows line up with the manifest
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    if progress:
        from tqdm import tqdm

        loader = tqdm(loader, desc="embedding", unit="batch")

    chunks = [model(batch.to(device)).float().cpu().numpy() for batch in loader]
    return np.concatenate(chunks, axis=0)
