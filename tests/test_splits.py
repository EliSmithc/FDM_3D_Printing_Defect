"""Tests for fold assignment.

A split bug is invisible at runtime and inflates every downstream score, so the
session-integrity guarantee is tested directly rather than trusted.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fdm_defect.splits import (
    assign_grouped_folds,
    assign_random_folds,
    audit_split,
    max_usable_splits,
)


def make_manifest(sessions_per_class: dict[str, list[int]]) -> pd.DataFrame:
    """Build a manifest from ``{label: [frames_in_session_1, ...]}``."""
    rows = []
    for label, sizes in sessions_per_class.items():
        for index, size in enumerate(sizes):
            session_id = f"{label}__s{index}"
            rows.extend(
                {"path": f"{session_id}/{n}.jpg", "label": label, "session_id": session_id}
                for n in range(size)
            )
    return pd.DataFrame(rows)


@pytest.fixture
def manifest():
    # Deliberately uneven, mirroring the real dataset: one class is rare and some
    # sessions are single-frame fragments.
    return make_manifest(
        {
            "Warping": [40, 30, 20, 10, 1],
            "Cracking": [35, 25, 15],
            "Off_platform": [30, 10, 3],
        }
    )


class TestMaxUsableSplits:
    def test_is_the_session_count_of_the_rarest_class(self, manifest):
        assert max_usable_splits(manifest) == 3


class TestGroupedFolds:
    def test_never_splits_a_session_across_folds(self, manifest):
        folds = assign_grouped_folds(manifest, n_splits=3)
        per_session = manifest.assign(fold=folds.to_numpy()).groupby("session_id")["fold"].nunique()
        assert (per_session == 1).all()

    def test_assigns_every_frame(self, manifest):
        folds = assign_grouped_folds(manifest, n_splits=3)
        assert (folds >= 0).all()
        assert len(folds) == len(manifest)

    def test_every_class_appears_in_every_fold(self, manifest):
        folds = assign_grouped_folds(manifest, n_splits=3)
        audit = audit_split(manifest, folds, strategy="grouped")
        assert audit.missing_class_folds == []

    def test_tolerates_single_frame_sessions(self, manifest):
        folds = assign_grouped_folds(manifest, n_splits=3)
        tiny = manifest["session_id"] == "Warping__s4"  # the 1-frame session
        assert folds[tiny].nunique() == 1

    def test_is_deterministic_for_a_seed(self, manifest):
        first = assign_grouped_folds(manifest, n_splits=3, seed=7)
        second = assign_grouped_folds(manifest, n_splits=3, seed=7)
        pd.testing.assert_series_equal(first, second)

    def test_rejects_more_folds_than_the_rarest_class_supports(self, manifest):
        with pytest.raises(ValueError, match="rarest class"):
            assign_grouped_folds(manifest, n_splits=4)


class TestRandomFolds:
    def test_splits_sessions_across_folds(self, manifest):
        """The leaky baseline must actually leak, or the comparison is meaningless."""
        folds = assign_random_folds(manifest, n_splits=3)
        audit = audit_split(manifest, folds, strategy="random")
        assert audit.leaks
        assert audit.leaked_frames > 0


class TestAuditSplit:
    def test_reports_a_clean_grouped_split(self, manifest):
        audit = audit_split(manifest, assign_grouped_folds(manifest, n_splits=3), "grouped")
        assert not audit.leaks
        assert audit.straddling_sessions == 0
        assert audit.leaked_frames == 0

    def test_counts_frames_that_would_have_to_move(self, manifest):
        # One session forced across two folds: the smaller side is the leak.
        small = make_manifest({"A": [10], "B": [10]})
        folds = pd.Series([0] * 7 + [1] * 3 + [0] * 10)
        audit = audit_split(small, folds, strategy="forced")
        assert audit.straddling_sessions == 1
        assert audit.leaked_frames == 3

    def test_flags_a_class_absent_from_a_fold(self):
        small = make_manifest({"A": [4], "B": [4]})
        folds = pd.Series([0] * 4 + [1] * 4)  # each class isolated in its own fold
        audit = audit_split(small, folds, strategy="forced")
        assert (0, "B") in audit.missing_class_folds
        assert (1, "A") in audit.missing_class_folds

    def test_describe_mentions_the_verdict(self, manifest):
        audit = audit_split(manifest, assign_grouped_folds(manifest, n_splits=3), "grouped")
        assert "clean" in audit.describe()
