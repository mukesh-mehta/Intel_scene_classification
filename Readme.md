# Intel scene classification — reproducible data preparation

[![Tests](https://github.com/mukesh-mehta/Intel_scene_classification/actions/workflows/ci.yml/badge.svg)](https://github.com/mukesh-mehta/Intel_scene_classification/actions/workflows/ci.yml)

Scene-classification experiments with a maintained tool for validated,
reproducible stratified folds. The historical image-model notebooks remain in
place; the new entry point makes their train/validation metadata dependable.
The repository's default branch remains **development**.

## Quick start

Use Python 3.12. The included metadata is synthetic and contains no images.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scene_data.py examples/metadata.csv artifacts/folds --folds 5 --seed 42
```

This writes `train_0.csv`/`val_0.csv` through `train_4.csv`/`val_4.csv`, plus
`manifest.json` recording the seed, class counts, fold count, and row totals.
Use a new or empty output directory for every experiment. On Windows activate
with `.venv\Scripts\activate`.

## Prepare your own data

Supply a CSV with `image_name,label` columns. Additional columns are preserved.
Image names must be unique, nonempty relative paths. Missing values, absolute
paths, `..` traversal, and single-class datasets are rejected. Each class must
have at least as many remaining rows as folds.

```csv
image_name,label
buildings/example.jpg,buildings
forest/example.jpg,forest
```

Use enough images for stratification; the two rows above only illustrate the
schema. Check files against an image root and reserve an untouched holdout:

```bash
python scene_data.py data/train.csv artifacts/experiment-1 \
  --image-root data/images --folds 5 --holdout 0.1 --seed 42
```

`--image-root` checks existence and containment, including resolved symlinks.
It does not decode images or detect identical image content under different names.
Holdout rows go to `train_dev.csv` and never appear in any fold. Each remaining
row appears in exactly one validation fold. The split uses shuffled stratification
and a fixed seed for both the holdout and folds.

The historical import `from utlities import stratified_split` still works; the
misspelled filename is retained for compatibility. Its original defaults remain
10 folds, 10% holdout, and seed 2. It now creates output directories, validates
inputs, and uses the same maintained splitter.

## Features and roadmap

| Capability | Status |
| --- | --- |
| CSV schema and missing-value checks | Implemented |
| Duplicate image-name rejection | Implemented |
| Optional local image existence checks | Implemented |
| Path containment checks | Implemented |
| Reproducible shuffled stratified folds | Implemented |
| Separate stratified holdout | Implemented |
| Manifest and notebook compatibility wrapper | Implemented |
| Split disjointness/reproducibility tests | Implemented |
| Image-content deduplication and corruption audit | Planned |
| Maintained training CLI with checkpoint resume | Planned |
| Group-aware splitting by source/location | Planned |
| Experiment comparison and inference UI | Planned |

## Historical notebooks

`av_torch*.ipynb` and `kfold_pytorch_transforms.ipynb` contain original PyTorch
experiments; `neural_network_forest.ipynb` explores combining outputs.
`tesing_on_cifar_data.ipynb` is a separate CIFAR experiment. Their dependencies,
data paths, training, and weights have not been migrated or reproduced here.
The original README reports private leaderboard rank 27/3013; this is historical,
not a result produced by the new splitter. See [original notes](docs/ORIGINAL_README.md).

Tests verify fold coverage, disjointness, holdout exclusion, repeatability, and
invalid metadata. No competition images or trained models are distributed.

Maintained code uses [MIT](LICENSE); historical notebooks/data retain their
existing terms. Read [NOTICE.md](NOTICE.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
