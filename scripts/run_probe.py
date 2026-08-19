"""Baseline: frozen backbone features plus a linear classifier.

Evaluates the same pipeline under both split strategies so the cost of leakage is
measured, not assumed. Embeddings are cached to disk on first run.

Usage:
    python scripts/run_probe.py [--model NAME] [--device mps|cuda|cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from fdm_defect.device import describe_device, resolve_device  # noqa: E402
from fdm_defect.embeddings import DEFAULT_MODEL, extract_embeddings, load_backbone  # noqa: E402
from fdm_defect.paths import ARTIFACTS_DIR, CACHE_DIR  # noqa: E402
from fdm_defect.splits import GROUPED, RANDOM  # noqa: E402


def load_or_extract(splits: pd.DataFrame, model_name: str, device_name: str | None) -> np.ndarray:
    cache_path = ARTIFACTS_DIR / f"embeddings_{model_name.replace('/', '_')}.npy"
    if cache_path.exists():
        embeddings = np.load(cache_path)
        if len(embeddings) == len(splits):
            print(f"Loaded cached embeddings {embeddings.shape} from {cache_path.name}")
            return embeddings
        print("Cached embeddings are stale; re-extracting.")

    device = resolve_device(device_name)
    print(f"Extracting with {model_name} on {describe_device(device)}")
    model, transform = load_backbone(model_name, device)

    image_root = CACHE_DIR / "512"
    paths = [image_root / path for path in splits["path"]]
    started = time.perf_counter()
    embeddings = extract_embeddings(paths, model, transform, device)
    print(f"  {embeddings.shape} in {time.perf_counter() - started:.0f}s")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def evaluate(embeddings: np.ndarray, splits: pd.DataFrame, fold_column: str) -> dict:
    """Cross-validated linear probe. Returns per-fold and pooled metrics."""
    labels = splits["label"].to_numpy()
    folds = splits[fold_column].to_numpy()

    predictions = np.empty_like(labels)
    fold_scores = []
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        classifier = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        scaler = StandardScaler().fit(embeddings[~test])
        classifier.fit(scaler.transform(embeddings[~test]), labels[~test])
        predictions[test] = classifier.predict(scaler.transform(embeddings[test]))
        fold_scores.append(f1_score(labels[test], predictions[test], average="macro"))

    frame_f1 = f1_score(labels, predictions, average="macro")

    # Aggregate to one prediction per print job so long jobs do not dominate.
    per_session = (
        pd.DataFrame({"session_id": splits["session_id"], "true": labels, "pred": predictions})
        .groupby("session_id")
        .agg(true=("true", "first"), pred=("pred", lambda s: s.mode().iloc[0]))
    )
    session_f1 = f1_score(per_session["true"], per_session["pred"], average="macro")

    return {
        "fold_macro_f1": [round(score, 4) for score in fold_scores],
        "frame_macro_f1": round(frame_f1, 4),
        "frame_macro_f1_std": round(float(np.std(fold_scores)), 4),
        "session_macro_f1": round(session_f1, 4),
        "session_accuracy": round(float((per_session["true"] == per_session["pred"]).mean()), 4),
        "report": classification_report(labels, predictions, digits=3, zero_division=0),
        "confusion": confusion_matrix(labels, predictions, labels=sorted(set(labels))),
        "classes": sorted(set(labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=ARTIFACTS_DIR / "splits.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.splits.exists():
        raise SystemExit(f"No splits at {args.splits}. Run scripts/build_splits.py first.")
    splits = pd.read_csv(args.splits)

    embeddings = load_or_extract(splits, args.model, args.device)

    results = {}
    for strategy in (GROUPED, RANDOM):
        print(f"\n{'=' * 62}\n{strategy.upper()} SPLIT\n{'=' * 62}")
        outcome = evaluate(embeddings, splits, f"fold_{strategy}")
        results[strategy] = outcome
        print(f"per-fold macro-F1 : {outcome['fold_macro_f1']}")
        print(
            f"frame macro-F1    : {outcome['frame_macro_f1']} (+/- {outcome['frame_macro_f1_std']})"
        )
        print(f"session macro-F1  : {outcome['session_macro_f1']}")
        print()
        print(outcome["report"])

    grouped_f1 = results[GROUPED]["frame_macro_f1"]
    random_f1 = results[RANDOM]["frame_macro_f1"]
    print(f"\n{'=' * 62}\nCOST OF LEAKAGE\n{'=' * 62}")
    print(f"  random split (leaky)  : {random_f1:.3f} macro-F1")
    print(f"  grouped split (honest): {grouped_f1:.3f} macro-F1")
    print(f"  inflation             : +{random_f1 - grouped_f1:.3f}")

    print("\nGrouped-split confusion (rows = true):")
    classes = results[GROUPED]["classes"]
    confusion = pd.DataFrame(results[GROUPED]["confusion"], index=classes, columns=classes)
    print(confusion.to_string())

    summary = {"model": args.model}
    for strategy, outcome in results.items():
        summary[strategy] = {
            key: value
            for key, value in outcome.items()
            if key not in {"report", "confusion", "classes"}
        }

    summary_path = ARTIFACTS_DIR / "probe_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
