"""Compatibility entry point for the original notebooks (filename retained)."""
from scene_data import create_folds


def stratified_split(train_file, path_to_save_metadata, keep_train_dev=True,
                     num_splits=10, random_state=2):
    return create_folds(train_file, path_to_save_metadata, folds=num_splits,
                        seed=random_state, holdout=0.1 if keep_train_dev else 0.0)
