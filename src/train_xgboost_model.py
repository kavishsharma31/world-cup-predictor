from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

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


def make_xgboost_pipeline(feature_columns: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                feature_columns,
            )
        ]
    )

    base_model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", base_model)])


def selected_feature_columns(df: pd.DataFrame, use_xg: bool) -> list[str]:
    if use_xg:
        return model_feature_columns(df)
    return [column for column in FEATURE_COLUMNS if column in df.columns]


def uses_xg_features(feature_columns: list[str]) -> bool:
    return any(column in XG_FEATURE_COLUMNS for column in feature_columns)


def write_metrics(metrics: dict, filename: str) -> None:
    with open(ARTIFACTS_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def train_xgboost_model(use_xg: bool = True) -> tuple[dict, dict]:
    """Train an XGBoost outcome classifier for comparison."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features().sort_values("date").reset_index(drop=True)
    feature_columns = selected_feature_columns(df, use_xg)
    train_df, test_df = split_train_test(df)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["result"])
    y_test = label_encoder.transform(test_df["result"])

    pipeline = make_xgboost_pipeline(feature_columns)
    pipeline.feature_columns_ = feature_columns
    pipeline.fit(train_df[feature_columns], y_train)

    pred = pipeline.predict(test_df[feature_columns])
    proba = pipeline.predict_proba(test_df[feature_columns])
    classes = list(label_encoder.classes_)

    metrics = {
        "model": "xgboost_outcome_classifier_xg" if use_xg else "xgboost_outcome_classifier_baseline",
        "training_cutoff_date": TRAINING_CUTOFF_DATE,
        "feature_end_date": str(df["date"].max().date()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_start_date": str(test_df["date"].min().date()),
        "test_end_date": str(test_df["date"].max().date()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "log_loss": float(log_loss(y_test, proba, labels=list(range(len(classes))))),
        "xg_features_used": uses_xg_features(feature_columns),
        "classes": classes,
        "feature_columns": feature_columns,
    }

    artifact = {"pipeline": pipeline, "label_encoder": label_encoder, "feature_columns": feature_columns}
    if use_xg:
        artifact_names = ["xgboost_outcome_model_xg.joblib", "xgboost_outcome_model.joblib"]
        metric_names = ["xgboost_xg_metrics.json", "xgboost_metrics.json"]
    else:
        artifact_names = ["xgboost_outcome_model_baseline.joblib"]
        metric_names = ["xgboost_baseline_metrics.json"]

    for artifact_name in artifact_names:
        joblib.dump(artifact, ARTIFACTS_DIR / artifact_name)
    for metric_name in metric_names:
        write_metrics(metrics, metric_name)

    return artifact, metrics


def train_xgboost_model_calibrated(use_xg: bool = True) -> tuple[dict, dict]:
    """Train a calibrated XGBoost outcome classifier."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features().sort_values("date").reset_index(drop=True)
    feature_columns = selected_feature_columns(df, use_xg)
    train_df, validation_df, test_df = split_train_validation_test(df)
    method = calibration_method(validation_df)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["result"])
    y_validation = label_encoder.transform(validation_df["result"])
    y_test = label_encoder.transform(test_df["result"])

    base_model = make_xgboost_pipeline(feature_columns)
    base_model.feature_columns_ = feature_columns
    base_model.fit(train_df[feature_columns], y_train)

    calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method=method)
    calibrated_model.fit(validation_df[feature_columns], y_validation)

    pred = calibrated_model.predict(test_df[feature_columns])
    proba = calibrated_model.predict_proba(test_df[feature_columns])
    classes = list(label_encoder.classes_)

    metrics = {
        "model": f"xgboost_outcome_classifier_{'xg' if use_xg else 'baseline'}_calibrated",
        "calibration_method": method,
        "training_cutoff_date": TRAINING_CUTOFF_DATE,
        "feature_end_date": str(df["date"].max().date()),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "test_start_date": str(test_df["date"].min().date()),
        "test_end_date": str(test_df["date"].max().date()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "log_loss": float(log_loss(y_test, proba, labels=list(range(len(classes))))),
        "brier_score": multiclass_brier_score(test_df["result"], proba, classes),
        "xg_features_used": uses_xg_features(feature_columns),
        "classes": classes,
        "feature_columns": feature_columns,
    }

    artifact = {"calibrated_model": calibrated_model, "label_encoder": label_encoder, "feature_columns": feature_columns}
    joblib.dump(artifact, ARTIFACTS_DIR / "xgboost_outcome_model_calibrated.joblib")
    with open(ARTIFACTS_DIR / "xgboost_calibrated_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return artifact, metrics


if __name__ == "__main__":
    _, metrics = train_xgboost_model()
    print(json.dumps(metrics, indent=2))
