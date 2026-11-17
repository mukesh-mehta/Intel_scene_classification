"""Validate scene metadata and create reproducible, leakage-free folds."""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def validate_metadata(frame, image_root=None):
    required = {"image_name", "label"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("CSV must contain nonempty image_name and label columns")
    if frame[list(required)].isna().any().any():
        raise ValueError("Image names and labels cannot be missing")
    if frame.image_name.duplicated().any():
        raise ValueError("Duplicate image names could leak between folds")
    root = Path(image_root).resolve() if image_root else None
    normalized_paths = set()
    for name in frame.image_name:
        if not isinstance(name, str) or not name.strip() or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Image names must be relative paths within the dataset")
        if root:
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Missing image or path outside image root: {name}")
        else:
            path = Path(name)
        if path in normalized_paths:
            raise ValueError("Duplicate image paths could leak between folds")
        normalized_paths.add(path)
    if frame.label.nunique() < 2:
        raise ValueError("At least two scene classes are required")
    return frame.reset_index(drop=True)


def create_folds(csv_path, output, *, folds=5, seed=42, holdout=0.0, image_root=None):
    frame = validate_metadata(pd.read_csv(csv_path, dtype={"image_name": str}), image_root)
    if not 0 <= holdout < 1 or folds < 2:
        raise ValueError("Require 0 <= holdout < 1 and at least two folds")
    reserved = None
    if holdout:
        frame, reserved = train_test_split(frame, test_size=holdout, stratify=frame.label, random_state=seed)
    frame = frame.reset_index(drop=True)
    if frame.label.value_counts().min() < folds:
        raise ValueError("Each class needs at least one row per fold after holdout")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty to avoid mixing experiments")
    output.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for i, (train, val) in enumerate(splitter.split(frame, frame.label)):
        frame.iloc[train].to_csv(output / f"train_{i}.csv", index=False)
        frame.iloc[val].to_csv(output / f"val_{i}.csv", index=False)
    if reserved is not None:
        reserved.to_csv(output / "train_dev.csv", index=False)
    manifest = {"seed": seed, "folds": folds, "rows": len(frame),
                "holdout_rows": 0 if reserved is None else len(reserved),
                "class_counts": {str(k): int(v) for k, v in frame.label.value_counts().items()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", type=float, default=0.0)
    parser.add_argument("--image-root", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(create_folds(args.csv, args.output, folds=args.folds, seed=args.seed,
                                      holdout=args.holdout, image_root=args.image_root), indent=2))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
