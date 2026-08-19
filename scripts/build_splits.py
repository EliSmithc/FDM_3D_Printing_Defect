"""Assign cross-validation folds and audit them for session leakage.

Writes artifacts/splits.csv with one fold column per strategy, and prints the audit
that justifies using the grouped split rather than the random one.

Usage:
    python scripts/build_splits.py [--n-splits 3] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from fdm_defect.manifest import training_rows  # noqa: E402
from fdm_defect.paths import ARTIFACTS_DIR, MANIFEST_PATH  # noqa: E402
from fdm_defect.splits import (  # noqa: E402
    DEFAULT_N_SPLITS,
    GROUPED,
    RANDOM,
    assign_grouped_folds,
    assign_random_folds,
    audit_split,
    max_usable_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "splits.csv")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = training_rows(pd.read_csv(args.manifest))
    print(f"{len(manifest)} frames across {manifest['session_id'].nunique()} sessions")
    print(f"rarest class has {max_usable_splits(manifest)} sessions -> max usable folds\n")

    strategies = {
        GROUPED: assign_grouped_folds(manifest, n_splits=args.n_splits, seed=args.seed),
        RANDOM: assign_random_folds(manifest, n_splits=args.n_splits, seed=args.seed),
    }

    splits = manifest[["path", "label", "session_id"]].copy()
    for name, folds in strategies.items():
        splits[f"fold_{name}"] = folds.to_numpy()
        print(audit_split(manifest, folds, strategy=name).describe())
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

    print("\nSessions per class per grouped test fold:")
    by_fold = splits.groupby(["fold_grouped", "label"])["session_id"]
    grouped = by_fold.nunique().unstack(fill_value=0)
    print(grouped.to_string())
    print("\nFrames per class per grouped test fold:")
    print(splits.groupby(["fold_grouped", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
