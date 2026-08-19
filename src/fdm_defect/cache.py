"""Building a downscaled working copy of the dataset.

The raw frames are 3072x2048 (~12 GB total). Decoding those on every epoch
dominates training time while adding no signal a 384px model can use, so we
decode once into a cached copy and iterate on that instead.

Uses Pillow's ``draft`` mode, which lets the JPEG decoder downscale by powers
of two while decoding rather than decoding at full size and resampling after.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

#: Length of the shorter edge in the cached copy. 512 leaves headroom to train
#: at 384 with random-resized-crop augmentation without upsampling.
DEFAULT_SHORT_SIDE = 512
DEFAULT_QUALITY = 92


@dataclass(frozen=True)
class CacheTask:
    source: Path
    destination: Path


def _resize_one(task: CacheTask, short_side: int, quality: int) -> bool:
    """Write one cached image. Returns True if work was done, False if skipped."""
    if task.destination.exists():
        return False

    task.destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(task.source) as image:
        # Ask the JPEG decoder for the smallest power-of-two reduction that
        # still exceeds our target, then resample the remainder precisely.
        image.draft("RGB", (short_side, short_side))
        image = image.convert("RGB")

        scale = short_side / min(image.size)
        if scale < 1:
            target = (round(image.width * scale), round(image.height * scale))
            image = image.resize(target, Image.Resampling.LANCZOS)

        # Write to a temporary name first so an interrupted run cannot leave a
        # truncated file that a later run would skip as already done.
        temporary = task.destination.with_suffix(".tmp.jpg")
        image.save(temporary, "JPEG", quality=quality, optimize=True)
        temporary.replace(task.destination)
    return True


def _worker(args: tuple[CacheTask, int, int]) -> bool:
    return _resize_one(*args)


def build_cache(
    tasks: Iterable[CacheTask],
    short_side: int = DEFAULT_SHORT_SIDE,
    quality: int = DEFAULT_QUALITY,
    workers: int | None = None,
    progress: bool = True,
) -> tuple[int, int]:
    """Resize every task in parallel. Returns ``(written, skipped)``."""
    payloads = [(task, short_side, quality) for task in tasks]
    if not payloads:
        return (0, 0)

    written = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(_worker, payloads, chunksize=8)
        if progress:
            from tqdm import tqdm

            results = tqdm(results, total=len(payloads), unit="img", desc="caching")
        written = sum(results)

    return written, len(payloads) - written
