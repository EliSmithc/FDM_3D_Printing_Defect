"""Recovering print-session structure from timelapse filenames.

The dataset ships as flat class folders, but the images are not independent
samples: they are frames captured every 30 seconds during a print job. Frames
from one job are near-duplicates of each other, so any split that mixes frames
from the same job across train and test leaks the answer.

Filenames encode a capture timestamp, e.g. ``Image_20231128195336980.jpg``
(``YYYYMMDDHHMMSS`` + milliseconds), optionally suffixed ``_aug`` or
``_original``. Sorting by timestamp within a class shows a cleanly bimodal gap
distribution: 30-second gaps inside a job, and gaps of many minutes between
jobs. ``SESSION_GAP_SECONDS`` sits in the empty region between the two modes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

#: Gap above which two consecutive frames are treated as separate print jobs.
#: Measured gaps are ~30s within a job and >15min between jobs, with a single
#: observed value anywhere in between, so this threshold is not sensitive.
SESSION_GAP_SECONDS = 15 * 60

#: ``Image_`` + 14 timestamp digits + 3 millisecond digits + optional variant.
_FILENAME_RE = re.compile(
    r"^Image_(?P<ts>\d{14})(?P<ms>\d{3})(?:_(?P<variant>aug|original))?\.jpg$",
    re.IGNORECASE,
)

#: ``base`` is the frame as captured. ``original`` is a lower-quality re-encode
#: of the same frame and ``aug`` is a horizontally flipped copy; both are
#: redundant copies the dataset authors bundled in, not new observations.
BASE = "base"
AUG = "aug"
ORIGINAL = "original"


@dataclass(frozen=True)
class Frame:
    """One parsed filename."""

    filename: str
    timestamp: datetime
    variant: str

    @property
    def capture_id(self) -> str:
        """Identifies the underlying capture, shared across all three variants."""
        return self.timestamp.strftime("%Y%m%d%H%M%S%f")[:17]

    @property
    def is_redundant_copy(self) -> bool:
        """True for the bundled duplicate variants that must not be trained on."""
        return self.variant != BASE


def parse_filename(filename: str) -> Frame | None:
    """Parse a dataset filename, or return None if it does not match the scheme."""
    match = _FILENAME_RE.match(filename)
    if match is None:
        return None
    stamp = f"{match['ts']}{match['ms']}"
    try:
        timestamp = datetime.strptime(stamp, "%Y%m%d%H%M%S%f")
    except ValueError:
        return None
    return Frame(
        filename=filename,
        timestamp=timestamp,
        variant=(match["variant"] or BASE).lower(),
    )


def parse_filenames(filenames: Iterable[str]) -> list[Frame]:
    """Parse many filenames, silently dropping any that do not match."""
    parsed = (parse_filename(name) for name in filenames)
    return [frame for frame in parsed if frame is not None]


def assign_sessions(
    frames: Sequence[Frame],
    label: str,
    gap_seconds: float = SESSION_GAP_SECONDS,
) -> dict[str, str]:
    """Group frames into print sessions, returning ``{filename: session_id}``.

    Session ids are derived from the first capture timestamp in the session
    (e.g. ``Warping__20231228-143327``) rather than a running index, so they
    stay stable if frames are later added or removed.
    """
    if not frames:
        return {}

    ordered = sorted(frames, key=lambda frame: (frame.timestamp, frame.filename))

    # Start a new session wherever the gap to the previous frame is too large.
    boundaries = [0]
    for index, (previous, current) in enumerate(zip(ordered, ordered[1:], strict=False), start=1):
        if (current.timestamp - previous.timestamp).total_seconds() > gap_seconds:
            boundaries.append(index)
    boundaries.append(len(ordered))

    assignments: dict[str, str] = {}
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        session = ordered[start:end]
        session_id = f"{label}__{session[0].timestamp:%Y%m%d-%H%M%S}"
        for frame in session:
            assignments[frame.filename] = session_id
    return assignments
