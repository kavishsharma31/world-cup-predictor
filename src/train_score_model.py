from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS_DIR, TRAINING_CUTOFF_DATE
from src.features import load_model_features, model_feature_columns, xg_features_used


def load_features() -> pd.DataFrame:
    return load_model_features()


def make_poisson_pipeline(feature_columns: list[str]) -> Pipeline:
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
    model = PoissonRegressor(alpha=0.01, max_iter=1000)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train_score_model() -> tuple[dict, dict]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features().sort_values("date").reset_index(drop=True)
    feature_columns = model_feature_columns(df)

    train_df = df[df["date"] < "2022-01-01"].copy()
    test_df = df[df["date"] >= "2022-01-01"].copy()
    if len(test_df) < 100:
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]

    home_model = make_poisson_pipeline(feature_columns)
    away_model = make_poisson_pipeline(feature_columns)

    home_model.fit(X_train, train_df["home_score"])
    away_model.fit(X_train, train_df["away_score"])

    home_pred = home_model.predict(X_test)
    away_pred = away_model.predict(X_test)

    metrics = {
        "model": "two_poisson_regressors",
        "training_cutoff_date": TRAINING_CUTOFF_DATE,
        "feature_end_date": str(df["date"].max().date()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_start_date": str(test_df["date"].min().date()),
        "test_end_date": str(test_df["date"].max().date()),
        "home_goals_mae": float(mean_absolute_error(test_df["home_score"], home_pred)),
        "away_goals_mae": float(mean_absolute_error(test_df["away_score"], away_pred)),
        "home_goals_rmse": float(mean_squared_error(test_df["home_score"], home_pred) ** 0.5),
        "away_goals_rmse": float(mean_squared_error(test_df["away_score"], away_pred) ** 0.5),
        "xg_features_used": xg_features_used(df),
        "feature_columns": feature_columns,
    }

    artifact = {"home_model": home_model, "away_model": away_model, "feature_columns": feature_columns}
    joblib.dump(artifact, ARTIFACTS_DIR / "score_model.joblib")
    with open(ARTIFACTS_DIR / "score_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return artifact, metrics


if __name__ == "__main__":
    _, metrics = train_score_model()
    print(json.dumps(metrics, indent=2))
