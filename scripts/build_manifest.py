"""Build the dataset manifest CSV.

Usage:
    python scripts/build_manifest.py [--no-hashes] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm_defect.manifest import build_manifest, training_rows  # noqa: E402
from fdm_defect.paths import MANIFEST_PATH, RAW_DATA_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--out", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--no-hashes",
        action="store_true",
        help="Skip md5 hashing (faster, but disables duplicate detection).",
    )
    args = parser.parse_args()

    manifest = build_manifest(data_dir=args.data_dir, compute_hashes=not args.no_hashes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)

    usable = training_rows(manifest)
    print(f"Wrote {len(manifest)} rows to {args.out}")
    print(f"  trainable frames : {len(usable)}")
    print(f"  redundant copies : {len(manifest) - len(usable)}")
    print(f"  print sessions   : {manifest['session_id'].nunique()}")
    print()
    summary = (
        usable.groupby("label")
        .agg(frames=("path", "size"), sessions=("session_id", "nunique"))
        .assign(frames_per_session=lambda df: (df["frames"] / df["sessions"]).round(1))
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
