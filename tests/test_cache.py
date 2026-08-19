"""Tests for the downscaled image cache."""

from __future__ import annotations

from PIL import Image

from fdm_defect.cache import CacheTask, build_cache


def make_source(path, size=(3072, 2048)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "grey").save(path, "JPEG")
    return path


class TestBuildCache:
    def test_downscales_the_short_side_preserving_aspect(self, tmp_path):
        task = CacheTask(make_source(tmp_path / "in/a.jpg"), tmp_path / "out/a.jpg")
        build_cache([task], short_side=512, progress=False)
        with Image.open(task.destination) as image:
            assert min(image.size) == 512
            assert image.size == (768, 512)

    def test_creates_nested_destination_directories(self, tmp_path):
        task = CacheTask(make_source(tmp_path / "in/a.jpg"), tmp_path / "out/Warping/a.jpg")
        build_cache([task], progress=False)
        assert task.destination.exists()

    def test_reports_written_and_skipped_counts(self, tmp_path):
        task = CacheTask(make_source(tmp_path / "in/a.jpg"), tmp_path / "out/a.jpg")
        assert build_cache([task], progress=False) == (1, 0)
        assert build_cache([task], progress=False) == (0, 1)

    def test_leaves_no_temporary_files_behind(self, tmp_path):
        task = CacheTask(make_source(tmp_path / "in/a.jpg"), tmp_path / "out/a.jpg")
        build_cache([task], progress=False)
        assert list((tmp_path / "out").glob("*.tmp.jpg")) == []

    def test_does_not_upscale_images_smaller_than_the_target(self, tmp_path):
        task = CacheTask(make_source(tmp_path / "in/a.jpg", (300, 200)), tmp_path / "out/a.jpg")
        build_cache([task], short_side=512, progress=False)
        with Image.open(task.destination) as image:
            assert image.size == (300, 200)

    def test_no_tasks_is_not_an_error(self, tmp_path):
        assert build_cache([], progress=False) == (0, 0)
