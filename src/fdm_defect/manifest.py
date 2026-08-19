"""Building the dataset manifest.

The manifest is the single source of truth every later stage reads: EDA, the
group-aware split, training and evaluation. Keeping it as one flat CSV means
the grouping logic is applied once, in one place, and is inspectable by hand.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from fdm_defect.paths import CLASSES, RAW_DATA_DIR
from fdm_defect.sessions import BASE, SESSION_GAP_SECONDS, assign_sessions, parse_filenames

MANIFEST_COLUMNS = [
    "path",
    "label",
    "session_id",
    "timestamp",
    "variant",
    "capture_id",
    "is_redundant_copy",
    "width",
    "height",
    "bytes",
    "md5",
]


@dataclass
class ManifestRow:
    path: str
    label: str
    session_id: str
    timestamp: str
    variant: str
    capture_id: str
    is_redundant_copy: bool
    width: int
    height: int
    bytes: int
    md5: str


def _md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()  # noqa: S324 - integrity/dedup only, not security
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    data_dir: Path = RAW_DATA_DIR,
    classes: tuple[str, ...] = CLASSES,
    gap_seconds: float = SESSION_GAP_SECONDS,
    compute_hashes: bool = True,
) -> pd.DataFrame:
    """Scan the class folders and return one row per image.

    Reads image headers for dimensions (cheap - Pillow does not decode pixels
    for ``.size``) and optionally hashes file contents so exact duplicates can
    be detected later.
    """
    rows: list[ManifestRow] = []

    for label in classes:
        class_dir = data_dir / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class directory at {class_dir}")

        frames = parse_filenames(entry.name for entry in class_dir.iterdir())
        sessions = assign_sessions(frames, label=label, gap_seconds=gap_seconds)

        for frame in frames:
            image_path = class_dir / frame.filename
            with Image.open(image_path) as image:
                width, height = image.size
            rows.append(
                ManifestRow(
                    path=str(image_path.relative_to(data_dir)),
                    label=label,
                    session_id=sessions[frame.filename],
                    timestamp=frame.timestamp.isoformat(),
                    variant=frame.variant,
                    capture_id=frame.capture_id,
                    is_redundant_copy=frame.is_redundant_copy,
                    width=width,
                    height=height,
                    bytes=image_path.stat().st_size,
                    md5=_md5(image_path) if compute_hashes else "",
                )
            )

    frame_table = pd.DataFrame([asdict(row) for row in rows], columns=MANIFEST_COLUMNS)
    return frame_table.sort_values(["label", "session_id", "timestamp", "variant"]).reset_index(
        drop=True
    )


def training_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    """The subset safe to train on: original captures only, no bundled copies."""
    return manifest.loc[manifest["variant"] == BASE].reset_index(drop=True)
