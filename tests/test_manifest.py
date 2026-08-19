"""Tests for manifest assembly, using a small synthetic dataset on disk."""

from __future__ import annotations

import pytest
from PIL import Image

from fdm_defect.manifest import MANIFEST_COLUMNS, build_manifest, training_rows


@pytest.fixture
def fake_dataset(tmp_path):
    """Two classes: one plain session, one session plus bundled variants."""
    layout = {
        "Cracking": [
            "Image_20231128195336980.jpg",
            "Image_20231128195406987.jpg",
            "Image_20231130101301000.jpg",  # two days later -> separate session
        ],
        "Warping": [
            "Image_20231228112532914.jpg",
            "Image_20231228112532914_aug.jpg",
            "Image_20231228112532914_original.jpg",
        ],
    }
    for label, filenames in layout.items():
        directory = tmp_path / label
        directory.mkdir()
        for filename in filenames:
            Image.new("RGB", (64, 32), "grey").save(directory / filename)
    (tmp_path / "Cracking" / ".DS_Store").write_bytes(b"junk")
    return tmp_path


class TestBuildManifest:
    def test_one_row_per_image_ignoring_junk(self, fake_dataset):
        manifest = build_manifest(fake_dataset, classes=("Cracking", "Warping"))
        assert len(manifest) == 6
        assert list(manifest.columns) == MANIFEST_COLUMNS

    def test_records_image_dimensions(self, fake_dataset):
        manifest = build_manifest(fake_dataset, classes=("Cracking",))
        assert set(manifest["width"]) == {64}
        assert set(manifest["height"]) == {32}

    def test_sessions_do_not_span_classes(self, fake_dataset):
        manifest = build_manifest(fake_dataset, classes=("Cracking", "Warping"))
        per_session = manifest.groupby("session_id")["label"].nunique()
        assert (per_session == 1).all()

    def test_splits_a_class_into_its_sessions(self, fake_dataset):
        manifest = build_manifest(fake_dataset, classes=("Cracking",))
        assert manifest["session_id"].nunique() == 2

    def test_paths_are_relative_to_the_data_directory(self, fake_dataset):
        manifest = build_manifest(fake_dataset, classes=("Cracking",))
        assert all(not path.startswith("/") for path in manifest["path"])
        assert all((fake_dataset / path).exists() for path in manifest["path"])

    def test_identical_files_share_an_md5(self, fake_dataset):
        manifest = build_manifest(fake_dataset, classes=("Warping",))
        assert manifest["md5"].nunique() == 1

    def test_hashing_can_be_skipped(self, fake_dataset):
        manifest = build_manifest(fake_dataset, classes=("Warping",), compute_hashes=False)
        assert (manifest["md5"] == "").all()

    def test_missing_class_directory_is_an_error(self, fake_dataset):
        with pytest.raises(FileNotFoundError):
            build_manifest(fake_dataset, classes=("Nonexistent",))


class TestTrainingRows:
    def test_excludes_the_bundled_copies(self, fake_dataset):
        manifest = build_manifest(fake_dataset, classes=("Cracking", "Warping"))
        usable = training_rows(manifest)
        assert len(usable) == 4
        assert (usable["variant"] == "base").all()
        assert not usable["is_redundant_copy"].any()
