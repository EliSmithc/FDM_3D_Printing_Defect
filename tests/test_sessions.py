"""Tests for print-session recovery.

This logic decides the train/test boundary, so a silent bug here produces
optimistic scores that look fine. It is worth testing directly.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from fdm_defect.sessions import (
    AUG,
    BASE,
    ORIGINAL,
    assign_sessions,
    parse_filename,
    parse_filenames,
)


def frame(stamp: str, variant: str = "") -> str:
    suffix = f"_{variant}" if variant else ""
    return f"Image_{stamp}{suffix}.jpg"


class TestParseFilename:
    def test_parses_a_plain_frame(self):
        parsed = parse_filename("Image_20231128195336980.jpg")
        assert parsed is not None
        assert parsed.timestamp == datetime(2023, 11, 28, 19, 53, 36, 980000)
        assert parsed.variant == BASE
        assert not parsed.is_redundant_copy

    @pytest.mark.parametrize(
        ("filename", "variant"),
        [
            ("Image_20231228112532914_aug.jpg", AUG),
            ("Image_20231228112532914_original.jpg", ORIGINAL),
        ],
    )
    def test_parses_bundled_variants(self, filename, variant):
        parsed = parse_filename(filename)
        assert parsed is not None
        assert parsed.variant == variant
        assert parsed.is_redundant_copy

    def test_variants_share_a_capture_id_with_their_base(self):
        stamp = "20231228112532914"
        ids = {parse_filename(frame(stamp, v)).capture_id for v in ("", AUG, ORIGINAL)}
        assert len(ids) == 1

    @pytest.mark.parametrize(
        "filename",
        [
            "notes.txt",
            "Image_2023112819533.jpg",  # too few digits
            "Image_20231328195336980.jpg",  # month 13
            "Image_20231128195336980_flipped.jpg",  # unknown variant
            ".DS_Store",
        ],
    )
    def test_rejects_anything_unexpected(self, filename):
        assert parse_filename(filename) is None

    def test_parse_filenames_drops_non_matching_entries(self):
        parsed = parse_filenames(["Image_20231128195336980.jpg", ".DS_Store", "readme.md"])
        assert len(parsed) == 1


class TestAssignSessions:
    def test_frames_30s_apart_form_one_session(self):
        frames = parse_filenames(
            [frame("20231128195336980"), frame("20231128195406987"), frame("20231128195436986")]
        )
        sessions = assign_sessions(frames, label="Cracking")
        assert len(set(sessions.values())) == 1

    def test_a_long_gap_starts_a_new_session(self):
        frames = parse_filenames([frame("20231130101301000"), frame("20231130111901000")])
        sessions = assign_sessions(frames, label="Cracking")
        assert len(set(sessions.values())) == 2

    def test_session_id_is_named_for_its_first_frame(self):
        frames = parse_filenames([frame("20231128195406987"), frame("20231128195336980")])
        sessions = assign_sessions(frames, label="Cracking")
        assert set(sessions.values()) == {"Cracking__20231128-195336"}

    def test_session_ids_are_stable_when_later_frames_are_removed(self):
        stamps = ["20231128195336980", "20231128195406987", "20231128195436986"]
        full = assign_sessions(parse_filenames([frame(s) for s in stamps]), label="Cracking")
        trimmed = assign_sessions(parse_filenames([frame(s) for s in stamps[:2]]), label="Cracking")
        assert trimmed[frame(stamps[0])] == full[frame(stamps[0])]

    def test_bundled_variants_land_in_their_base_frames_session(self):
        stamp = "20231228112532914"
        frames = parse_filenames([frame(stamp), frame(stamp, AUG), frame(stamp, ORIGINAL)])
        sessions = assign_sessions(frames, label="Warping")
        assert len(set(sessions.values())) == 1

    def test_every_frame_is_assigned(self):
        frames = parse_filenames([frame("20231128195336980"), frame("20240118164355000")])
        sessions = assign_sessions(frames, label="Cracking")
        assert len(sessions) == len(frames)

    def test_no_frames_yields_no_sessions(self):
        assert assign_sessions([], label="Cracking") == {}

    def test_gap_threshold_is_configurable(self):
        frames = parse_filenames([frame("20231128195336980"), frame("20231128195406987")])
        assert len(set(assign_sessions(frames, "Cracking", gap_seconds=10).values())) == 2
        assert len(set(assign_sessions(frames, "Cracking", gap_seconds=60).values())) == 1
