"""Decode the raw frames once into a downscaled cache used for training.

Usage:
    python scripts/build_cache.py [--short-side 512] [--workers N] [--all-variants]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from fdm_defect.cache import (  # noqa: E402
    DEFAULT_QUALITY,
    DEFAULT_SHORT_SIDE,
    CacheTask,
    build_cache,
)
from fdm_defect.manifest import training_rows  # noqa: E402
from fdm_defect.paths import CACHE_DIR, MANIFEST_PATH, RAW_DATA_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--short-side", type=int, default=DEFAULT_SHORT_SIDE)
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Also cache the bundled _aug/_original copies (excluded by default).",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"No manifest at {args.manifest}. Run scripts/build_manifest.py first.")

    manifest = pd.read_csv(args.manifest)
    rows = manifest if args.all_variants else training_rows(manifest)

    destination_root = args.cache_dir / str(args.short_side)
    tasks = [
        CacheTask(source=args.data_dir / row.path, destination=destination_root / row.path)
        for row in rows.itertuples()
    ]

    written, skipped = build_cache(
        tasks, short_side=args.short_side, quality=args.quality, workers=args.workers
    )

    total_bytes = sum(p.stat().st_size for p in destination_root.rglob("*.jpg"))
    print(f"\nCache at {destination_root}")
    print(f"  written : {written}")
    print(f"  skipped : {skipped} (already cached)")
    print(f"  size    : {total_bytes / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
