from __future__ import annotations

import numpy as np
import pandas as pd


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the existing time-based test split."""
    df = df.sort_values("date").reset_index(drop=True)
    train_df = df[df["date"] < "2022-01-01"].copy()
    test_df = df[df["date"] >= "2022-01-01"].copy()

    if len(test_df) < 100:
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def split_train_validation_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split without using test rows for calibration."""
    train_pool, test_df = split_train_test(df)
    train_df = train_pool[train_pool["date"] < "2021-01-01"].copy()
    validation_df = train_pool[train_pool["date"] >= "2021-01-01"].copy()

    if len(validation_df) < 100 or validation_df["result"].nunique() < 3:
        split_idx = int(len(train_pool) * 0.8)
        train_df = train_pool.iloc[:split_idx].copy()
        validation_df = train_pool.iloc[split_idx:].copy()

    return train_df, validation_df, test_df


def calibration_method(validation_df: pd.DataFrame) -> str:
    """Use isotonic only when the validation set is large enough."""
    class_counts = validation_df["result"].value_counts()
    if len(validation_df) >= 1000 and validation_df["result"].nunique() >= 3 and class_counts.min() >= 20:
        return "isotonic"
    return "sigmoid"


def multiclass_brier_score(y_true: pd.Series, pred_proba: np.ndarray, classes: list) -> float:
    class_to_index = {label: i for i, label in enumerate(classes)}
    y = np.zeros((len(y_true), len(classes)))
    for row_idx, label in enumerate(y_true):
        y[row_idx, class_to_index[label]] = 1.0
    return float(np.mean(np.sum((pred_proba - y) ** 2, axis=1)))
