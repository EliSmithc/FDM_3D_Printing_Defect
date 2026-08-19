# Detecting 3D Print Failures from Webcam Frames

A computer vision project that classifies FDM 3D printing defects — cracking, layer
shifting, parts coming off the platform, stringing, and warping — from photos taken by a
printer's camera.

The interesting part of this project turned out not to be the model. It was discovering
that the dataset contains far less information than its file count suggests, and building
an evaluation that doesn't lie about it.

---

## The dataset size is deceiving

The [FDM 3D Printing Defect Dataset](https://www.kaggle.com/datasets/wengmhu/fdm-3d-printing-defect-dataset)
sorts 1,912 photos sorted into five classes.

The photos are **timelapse frames**, captured every 30 seconds
while a print was running. A single print job contributes around 50 photos of the same
object, from the same fixed camera, under the same lighting — all labelled identically,
because the label describes how *that print* failed.

Reconstructing job boundaries from the timestamps embedded in the filenames gives the
number of print jobs per failure mode:

| Defect | Photos | Print jobs | Photos per job |
|---|---:|---:|---|
| Cracking | 430 | 8 | 2 – 85 |
| Layer shifting | 329 | 5 | 47 – 79 |
| Off platform | 91 | 3 | 3 – 66 |
| Stringing | 419 | 9 | 24 – 88 |
| Warping | 482 | 15 | 1 – 75 |
| **Total** | **1,751** | **40** | **median 46** |

Job sizes vary widely. Most run 30–80 photos, but a handful are isolated fragments of
one or two frames captured well away from any other. The ten largest jobs hold 44% of
every photo in the dataset.

## Why it matters

Split those 1,912 photos randomly into training and test sets, and frames from the same
print job land on both sides. The model gets tested on a photo of an object it may have already seen.

It doesn't need to learn what the defect looks like to pass that test. It only needs to
recognise *that particular print*.

Every split in this project keeps whole print jobs together. Train on some jobs, test
on others.

## Other things worth knowing about the data

**161 files are duplicates the dataset authors left in.** Scattered through the class
folders are files ending `_original` and `_aug`. Comparing them pixel by pixel against
their source frames: `_original` is just a lower-quality re-encode of a photo already
present, and `_aug` is a mix of 40 horizontal flips, 23 brightness and colour shifts, and
18 files that aren't transformed at all. They're pre-baked augmentation, they add nothing
a training pipeline can't do on the fly, and they'd leak across splits. Training uses the
1,751 genuine frames only.

**The classes are lopsided.** Warping has 482 photos across 15 jobs; off-platform has 91
across just 3, sized 66, 22 and 3 photos. Hold out the largest of those for testing and
the class is left training on 25 photos. Any off-platform result deserves a large asterisk.

**Job sizes skew the scoring.** Because some jobs contributed three times as many photos
as others, a per-photo score quietly counts the biggest jobs most. Results are therefore
also reported per print job, so each job counts once no matter how long it ran.

---

## First results

A deliberately simple baseline: run a frozen [DINOv2](https://arxiv.org/abs/2304.07193)
vision transformer over every photo once, then fit logistic regression on the resulting
feature vectors. No training loop, no fine-tuning — about a minute end to end. The point
is to establish what's achievable cheaply before building anything larger.

The same pipeline was then scored twice, once under each splitting strategy:

| Split | Macro-F1 |
|---|---:|
| Random (frames shuffled) | **0.991** |
| Grouped (whole print jobs held out) | **0.783** |

That gap is the entire argument of this project made concrete. The random split scatters
61% of all photos into a fold containing a near-duplicate neighbour, and the resulting
0.991 is measuring memorisation. The honest number is 0.783.

Per-defect, on the grouped split:

| Defect | F1 | |
|---|---:|---|
| Stringing | 0.988 | essentially solved |
| Warping | 0.864 | |
| Off platform | 0.828 | only 3 jobs — treat with caution |
| Cracking | 0.635 | |
| Layer shifting | 0.599 | |

**Cracking and layer shifting are the real problem.** They account for most of the
errors, and they're confused with each other in both directions — 142 cracking photos
called layer shifting, 99 the reverse. That is plausible: both produce a horizontal
discontinuity in the side of a print, and telling them apart may need finer detail than
a frozen backbone preserves, or may be genuinely ambiguous in some frames.

Scores also vary a lot by fold (0.64, 0.90, 0.65). With only 40 print jobs, which jobs
land in the test set matters enormously — another reason to distrust any single number
from this dataset, including these.

---

## Running it

Requires Python 3.11+ and the dataset downloaded from Kaggle into
`dataset/FDM-3D-Printing-Defect-Dataset/data/`.

Dependencies are managed with [uv](https://docs.astral.sh/uv/). `uv sync` reads
`uv.lock` and builds an environment with the exact versions these results came from.

```bash
uv sync                                  # create .venv from the lockfile

uv run scripts/build_manifest.py         # index every photo, recover print jobs
uv run scripts/build_cache.py            # shrink 11 GB of photos to 88 MB
uv run scripts/build_splits.py           # assign folds, audit them for leakage
uv run scripts/run_probe.py              # baseline results
```

The first script writes `artifacts/manifest.csv`, one row per photo, recording which
print job it belongs to. Everything downstream reads from it, so the grouping logic
lives in exactly one place.

The second decodes the 3072×2048 originals once into a smaller working copy. Training
never needs that resolution, and re-decoding 11 GB every epoch would dominate runtime.
It takes about five seconds.

Tests and linting:

```bash
uv run pytest
uv run ruff check .
```

## Where the project is

- [x] Recovering print-job structure from filename timestamps
- [x] Dataset manifest and duplicate audit
- [x] Downscaled image cache
- [x] Job-grouped train/test split, with a leakage audit
- [x] Baseline: frozen vision backbone plus logistic regression
- [ ] Fine-tuned classifier
- [ ] Evaluation, error analysis, and Grad-CAM attention maps
- [ ] Model card
