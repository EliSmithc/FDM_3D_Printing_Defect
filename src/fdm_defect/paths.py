"""Canonical project paths.

Everything is resolved relative to the repository root so that scripts behave
the same regardless of the working directory they are invoked from.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "FDM-3D-Printing-Defect-Dataset" / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MANIFEST_PATH = ARTIFACTS_DIR / "manifest.csv"
CACHE_DIR = PROJECT_ROOT / "dataset" / "cache"

CLASSES = (
    "Cracking",
    "Layer_shifting",
    "Off_platform",
    "Stringing",
    "Warping",
)
