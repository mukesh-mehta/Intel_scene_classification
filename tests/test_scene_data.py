import pandas as pd
import pytest
from scene_data import create_folds, validate_metadata


def data():
    return pd.DataFrame({"image_name": [f"{i}.jpg" for i in range(60)], "label": [i % 3 for i in range(60)]})


def test_folds_reproducible_disjoint_and_complete(tmp_path):
    csv = tmp_path / "metadata.csv"
    data().to_csv(csv, index=False)
    for name in ["a", "b"]:
        manifest = create_folds(csv, tmp_path / name, folds=3, holdout=0.2)
        assert manifest["holdout_rows"] == 12
    seen = set()
    reserved = set(pd.read_csv(tmp_path / "a/train_dev.csv").image_name)
    for i in range(3):
        train = pd.read_csv(tmp_path / f"a/train_{i}.csv")
        val = pd.read_csv(tmp_path / f"a/val_{i}.csv")
        assert not set(train.image_name) & set(val.image_name)
        assert not reserved & (set(train.image_name) | set(val.image_name))
        assert not seen & set(val.image_name)
        seen.update(val.image_name)
        counts = val.label.value_counts()
        assert counts.max() - counts.min() <= 1
        assert (tmp_path / f"a/val_{i}.csv").read_bytes() == (tmp_path / f"b/val_{i}.csv").read_bytes()
    assert len(seen | reserved) == 60
    with pytest.raises(ValueError, match="empty"):
        create_folds(csv, tmp_path / "a", folds=3)


@pytest.mark.parametrize("name", ["../outside.jpg", "/absolute.jpg", "", None])
def test_invalid_image_names(name):
    frame = data()
    frame.loc[0, "image_name"] = name
    with pytest.raises(ValueError):
        validate_metadata(frame)


def test_duplicate_missing_and_undersized(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        validate_metadata(pd.concat([data(), data()]))
    with pytest.raises(ValueError, match="Missing image"):
        validate_metadata(data(), tmp_path)
    csv = tmp_path / "data.csv"
    data().head(6).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Each class"):
        create_folds(csv, tmp_path / "out", folds=3)
    assert not (tmp_path / "out").exists()


def test_equivalent_paths_are_duplicates():
    frame = data()
    frame.loc[1, "image_name"] = "./0.jpg"
    with pytest.raises(ValueError, match="Duplicate image paths"):
        validate_metadata(frame)
