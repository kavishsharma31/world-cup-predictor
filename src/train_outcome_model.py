from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS_DIR, FEATURE_COLUMNS, RANDOM_STATE, TRAINING_CUTOFF_DATE, XG_FEATURE_COLUMNS
from src.features import load_model_features, model_feature_columns
from src.model_utils import (
    calibration_method,
    multiclass_brier_score,
    split_train_test,
    split_train_validation_test,
)


def load_features() -> pd.DataFrame:
    return load_model_features()


def make_logistic_pipeline(feature_columns: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_columns)],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def selected_feature_columns(df: pd.DataFrame, use_xg: bool) -> list[str]:
    if use_xg:
        return model_feature_columns(df)
    return [column for column in FEATURE_COLUMNS if column in df.columns]


def uses_xg_features(feature_columns: list[str]) -> bool:
    return any(column in XG_FEATURE_COLUMNS for column in feature_columns)


def write_metrics(metrics: dict, filename: str) -> None:
    with open(ARTIFACTS_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def train_outcome_model(use_xg: bool = True) -> tuple[Pipeline, dict]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features().sort_values("date").reset_index(drop=True)
    feature_columns = selected_feature_columns(df, use_xg)
    train_df, test_df = split_train_test(df)

    X_train = train_df[feature_columns]
    y_train = train_df["result"]
    X_test = test_df[feature_columns]
    y_test = test_df["result"]

    pipeline = make_logistic_pipeline(feature_columns)
    pipeline.feature_columns_ = feature_columns
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    pred_proba = pipeline.predict_proba(X_test)

    metrics = {
        "model": "multinomial_logistic_regression",
        "training_cutoff_date": TRAINING_CUTOFF_DATE,
        "feature_end_date": str(df["date"].max().date()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_start_date": str(test_df["date"].min().date()),
        "test_end_date": str(test_df["date"].max().date()),
        "accuracy": float(accuracy_score(y_test, preds)),
        "log_loss": float(log_loss(y_test, pred_proba, labels=list(pipeline.classes_))),
        "xg_features_used": uses_xg_features(feature_columns),
        "classes": list(pipeline.classes_),
        "classification_report": classification_report(y_test, preds, output_dict=True),
        "feature_columns": feature_columns,
    }

    if use_xg:
        metrics["model"] = "multinomial_logistic_regression_xg"
        artifact_names = ["outcome_model_xg.joblib", "outcome_model.joblib"]
        metric_names = ["outcome_xg_metrics.json", "outcome_metrics.json"]
    else:
        metrics["model"] = "multinomial_logistic_regression_baseline"
        artifact_names = ["outcome_model_baseline.joblib"]
        metric_names = ["outcome_baseline_metrics.json"]

    for artifact_name in artifact_names:
        joblib.dump(pipeline, ARTIFACTS_DIR / artifact_name)
    for metric_name in metric_names:
        write_metrics(metrics, metric_name)

    return pipeline, metrics


def train_outcome_model_calibrated(use_xg: bool = True) -> tuple[CalibratedClassifierCV, dict]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features().sort_values("date").reset_index(drop=True)
    feature_columns = selected_feature_columns(df, use_xg)
    train_df, validation_df, test_df = split_train_validation_test(df)
    method = calibration_method(validation_df)

    base_model = make_logistic_pipeline(feature_columns)
    base_model.fit(train_df[feature_columns], train_df["result"])

    calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method=method)
    calibrated_model.feature_columns_ = feature_columns
    calibrated_model.fit(validation_df[feature_columns], validation_df["result"])

    preds = calibrated_model.predict(test_df[feature_columns])
    pred_proba = calibrated_model.predict_proba(test_df[feature_columns])
    classes = list(calibrated_model.classes_)

    metrics = {
        "model": "multinomial_logistic_regression_calibrated",
        "calibration_method": method,
        "training_cutoff_date": TRAINING_CUTOFF_DATE,
        "feature_end_date": str(df["date"].max().date()),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "test_start_date": str(test_df["date"].min().date()),
        "test_end_date": str(test_df["date"].max().date()),
        "accuracy": float(accuracy_score(test_df["result"], preds)),
        "log_loss": float(log_loss(test_df["result"], pred_proba, labels=classes)),
        "brier_score": multiclass_brier_score(test_df["result"], pred_proba, classes),
        "xg_features_used": uses_xg_features(feature_columns),
        "classes": classes,
        "feature_columns": feature_columns,
    }

    suffix = "xg" if use_xg else "baseline"
    metrics["model"] = f"multinomial_logistic_regression_{suffix}_calibrated"
    joblib.dump(calibrated_model, ARTIFACTS_DIR / "outcome_model_calibrated.joblib")
    write_metrics(metrics, "outcome_calibrated_metrics.json")

    return calibrated_model, metrics


if __name__ == "__main__":
    _, metrics = train_outcome_model()
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
