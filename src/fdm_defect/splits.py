"""Assigning frames to cross-validation folds without leaking print sessions.

Every frame of a print session shares one label and looks nearly identical to its
neighbours, so a session must land entirely inside one fold. Anything else lets the
model recognise the session instead of the defect.

The fold count is capped by the rarest class: Off_platform has only 3 sessions, so
``n_splits`` above 3 leaves some test folds with no Off_platform frames at all and its
per-class recall becomes undefined there. Hence the default of 3.

``assign_random_folds`` deliberately implements the *wrong* split - shuffling frames
with no regard for sessions - so the leakage it causes can be measured rather than
asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

#: Highest fold count that still puts at least one session of every class in each test
#: fold, given Off_platform's 3 sessions.
DEFAULT_N_SPLITS = 3

GROUPED = "grouped"
RANDOM = "random"


def max_usable_splits(manifest: pd.DataFrame) -> int:
    """Largest fold count where every class can still appear in every test fold."""
    return int(manifest.groupby("label")["session_id"].nunique().min())


def assign_grouped_folds(
    manifest: pd.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = 0,
) -> pd.Series:
    """Fold index per row, keeping whole sessions together and classes balanced."""
    limit = max_usable_splits(manifest)
    if n_splits > limit:
        raise ValueError(
            f"n_splits={n_splits} exceeds {limit}, the number of sessions in the rarest "
            "class; some test folds would contain none of it."
        )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = pd.Series(-1, index=manifest.index, dtype=int)
    for fold, (_, test_index) in enumerate(
        splitter.split(manifest, y=manifest["label"], groups=manifest["session_id"])
    ):
        folds.iloc[test_index] = fold
    return folds


def assign_random_folds(
    manifest: pd.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = 0,
) -> pd.Series:
    """Fold index per row ignoring sessions entirely - the leaky split, for comparison."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = pd.Series(-1, index=manifest.index, dtype=int)
    for fold, (_, test_index) in enumerate(splitter.split(manifest, y=manifest["label"])):
        folds.iloc[test_index] = fold
    return folds


@dataclass
class SplitAudit:
    """Evidence about whether a fold assignment leaks."""

    strategy: str
    n_splits: int
    straddling_sessions: int
    leaked_frames: int
    missing_class_folds: list[tuple[int, str]]
    fold_sizes: dict[int, int]

    @property
    def leaks(self) -> bool:
        return self.straddling_sessions > 0

    def describe(self) -> str:
        lines = [
            f"strategy         : {self.strategy}",
            f"folds            : {self.n_splits}",
            f"fold sizes       : {dict(sorted(self.fold_sizes.items()))}",
            f"sessions split   : {self.straddling_sessions}",
            f"frames leaked    : {self.leaked_frames}",
        ]
        if self.missing_class_folds:
            absent = ", ".join(f"fold {f}: {label}" for f, label in self.missing_class_folds)
            lines.append(f"classes missing  : {absent}")
        lines.append(f"verdict          : {'LEAKS' if self.leaks else 'clean'}")
        return "\n".join(lines)


def audit_split(manifest: pd.DataFrame, folds: pd.Series, strategy: str) -> SplitAudit:
    """Check a fold assignment for sessions that straddle folds and absent classes.

    A session that appears in more than one fold means near-duplicate frames sit on both
    sides of the train/test boundary. ``leaked_frames`` counts the frames that would have
    to move to repair it.
    """
    table = manifest.assign(fold=folds.to_numpy())

    per_session = table.groupby("session_id")["fold"]
    straddling = per_session.nunique()
    leaked = 0
    for session_id in straddling[straddling > 1].index:
        counts = table.loc[table["session_id"] == session_id, "fold"].value_counts()
        leaked += int(counts.iloc[1:].sum())  # all but the session's majority fold

    present = set(zip(table["fold"], table["label"], strict=True))
    missing = [
        (fold, label)
        for fold in sorted(table["fold"].unique())
        for label in sorted(manifest["label"].unique())
        if (fold, label) not in present
    ]

    return SplitAudit(
        strategy=strategy,
        n_splits=int(table["fold"].nunique()),
        straddling_sessions=int((straddling > 1).sum()),
        leaked_frames=leaked,
        missing_class_folds=missing,
        fold_sizes=table["fold"].value_counts().to_dict(),
    )
