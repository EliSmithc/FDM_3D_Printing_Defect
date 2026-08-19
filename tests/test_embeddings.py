"""Tests for embedding extraction plumbing.

Deliberately avoids downloading pretrained weights; what matters here is that image
loading and ordering are correct, since a silent reorder would misalign every row
against the manifest.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from fdm_defect.embeddings import ImageListDataset, extract_embeddings


def to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)


def write_images(tmp_path, count, size=(8, 8)):
    paths = []
    for index in range(count):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", size, (index * 10, 0, 0)).save(path)
        paths.append(path)
    return paths


class TestImageListDataset:
    def test_length_matches_the_path_list(self, tmp_path):
        dataset = ImageListDataset(write_images(tmp_path, 3), to_tensor)
        assert len(dataset) == 3

    def test_converts_greyscale_sources_to_three_channels(self, tmp_path):
        path = tmp_path / "grey.jpg"
        Image.new("L", (8, 8), 128).save(path)
        assert ImageListDataset([path], to_tensor)[0].shape[0] == 3


class TestExtractEmbeddings:
    def test_preserves_input_order(self, tmp_path):
        """Row i of the output must correspond to path i, or labels misalign."""
        paths = write_images(tmp_path, 6)

        def mean_brightness(batch):
            return batch.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)

        embeddings = extract_embeddings(
            paths,
            mean_brightness,
            to_tensor,
            torch.device("cpu"),
            batch_size=2,
            num_workers=0,
            progress=False,
        )
        assert embeddings.shape == (6, 1)
        assert np.all(np.diff(embeddings[:, 0]) > 0)  # brightness increases by construction

    def test_returns_one_row_per_image(self, tmp_path):
        paths = write_images(tmp_path, 5)
        embeddings = extract_embeddings(
            paths,
            lambda batch: batch.flatten(1),
            to_tensor,
            torch.device("cpu"),
            batch_size=2,
            num_workers=0,
            progress=False,
        )
        assert len(embeddings) == len(paths)
