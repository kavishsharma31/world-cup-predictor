from __future__ import annotations

import joblib
import pandas as pd

from src.config import ARTIFACTS_DIR, FEATURE_COLUMNS, PROCESSED_DIR, XG_FEATURE_COLUMNS
from src.score_utils import outcome_from_scorelines, scoreline_probabilities

OUTCOME_MODEL_LABELS = {
    "Logistic Regression Baseline": "outcome_model_baseline.joblib",
    "Logistic Regression + xG": "outcome_model_xg.joblib",
    "Logistic Regression + xG Calibrated": "outcome_model_calibrated.joblib",
    "XGBoost Baseline": "xgboost_outcome_model_baseline.joblib",
    "XGBoost + xG": "xgboost_outcome_model_xg.joblib",
    "XGBoost + xG Calibrated": "xgboost_outcome_model_calibrated.joblib",
}
LEGACY_OUTCOME_MODEL_LABELS = {
    "Logistic Regression": "outcome_model.joblib",
    "Logistic Regression Calibrated": "outcome_model_calibrated.joblib",
    "XGBoost": "xgboost_outcome_model.joblib",
    "XGBoost Calibrated": "xgboost_outcome_model_calibrated.joblib",
}
LATEST_XG_COLUMNS = [
    "xg_for_avg_last_5",
    "xg_against_avg_last_5",
    "non_penalty_xg_for_avg_last_5",
    "non_penalty_xg_against_avg_last_5",
]


def load_latest_team_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "latest_team_features.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python scripts/run_full_pipeline.py` first.")
    return pd.read_csv(path, parse_dates=["last_match_date"])


def make_fixture_features(
    home_team: str,
    away_team: str,
    latest_team_features: pd.DataFrame,
    neutral: int = 1,
    tournament: str = "FIFA World Cup",
) -> pd.DataFrame:
    """Create a one-row feature table for a future fixture."""
    if home_team == away_team:
        raise ValueError("home_team and away_team must be different.")

    latest = latest_team_features.set_index("team")
    if home_team not in latest.index:
        raise ValueError(f"Unknown home team: {home_team}")
    if away_team not in latest.index:
        raise ValueError(f"Unknown away team: {away_team}")

    h = latest.loc[home_team]
    a = latest.loc[away_team]
    tournament_lower = tournament.lower()

    row = {
        "elo_diff": float(h["current_elo"] - a["current_elo"]),
        "neutral": int(neutral),
        "is_world_cup": int(tournament_lower == "fifa world cup"),
        "is_qualifier": int("qualification" in tournament_lower),
        "is_friendly": int("friendly" in tournament_lower),
    }

    for window in (5, 10):
        for metric in [
            "points_per_game",
            "goal_diff_avg",
            "win_rate",
            "goals_for_avg",
            "goals_against_avg",
        ]:
            h_col = f"{metric}_last_{window}"
            a_col = f"{metric}_last_{window}"
            row[f"home_{metric}_last_{window}"] = float(h.get(h_col, 0.0))
            row[f"away_{metric}_last_{window}"] = float(a.get(a_col, 0.0))

        row[f"points_per_game_diff_last_{window}"] = (
            row[f"home_points_per_game_last_{window}"] - row[f"away_points_per_game_last_{window}"]
        )
        row[f"goal_diff_avg_diff_last_{window}"] = (
            row[f"home_goal_diff_avg_last_{window}"] - row[f"away_goal_diff_avg_last_{window}"]
        )
        row[f"win_rate_diff_last_{window}"] = row[f"home_win_rate_last_{window}"] - row[f"away_win_rate_last_{window}"]
        row[f"goals_for_avg_diff_last_{window}"] = (
            row[f"home_goals_for_avg_last_{window}"] - row[f"away_goals_for_avg_last_{window}"]
        )
        row[f"goals_against_avg_diff_last_{window}"] = (
            row[f"home_goals_against_avg_last_{window}"] - row[f"away_goals_against_avg_last_{window}"]
        )

    if all(column in latest_team_features.columns for column in LATEST_XG_COLUMNS):
        for metric in LATEST_XG_COLUMNS:
            row[f"home_{metric}"] = float(h.get(metric, 0.0))
            row[f"away_{metric}"] = float(a.get(metric, 0.0))
        row["xg_for_avg_diff_last_5"] = row["home_xg_for_avg_last_5"] - row["away_xg_for_avg_last_5"]
        row["xg_against_avg_diff_last_5"] = (
            row["home_xg_against_avg_last_5"] - row["away_xg_against_avg_last_5"]
        )
        row["non_penalty_xg_for_avg_diff_last_5"] = (
            row["home_non_penalty_xg_for_avg_last_5"] - row["away_non_penalty_xg_for_avg_last_5"]
        )
        row["non_penalty_xg_against_avg_diff_last_5"] = (
            row["home_non_penalty_xg_against_avg_last_5"] - row["away_non_penalty_xg_against_avg_last_5"]
        )

    for col in FEATURE_COLUMNS:
        row.setdefault(col, 0.0)
    feature_columns = FEATURE_COLUMNS + [column for column in XG_FEATURE_COLUMNS if column in row]

    return pd.DataFrame([row])[feature_columns]


def load_outcome_artifact(model_name: str = "Logistic Regression + xG"):
    model_labels = {**OUTCOME_MODEL_LABELS, **LEGACY_OUTCOME_MODEL_LABELS}
    if model_name not in model_labels:
        raise ValueError(f"Unknown outcome model: {model_name}")
    return joblib.load(ARTIFACTS_DIR / model_labels[model_name])


def artifact_feature_columns(artifact) -> list[str] | None:
    if isinstance(artifact, dict):
        return artifact.get("feature_columns")
    feature_columns = getattr(artifact, "feature_columns_", None)
    if feature_columns is not None:
        return list(feature_columns)
    preprocessor = getattr(artifact, "named_steps", {}).get("preprocess") if hasattr(artifact, "named_steps") else None
    if preprocessor is not None and getattr(preprocessor, "transformers", None):
        return list(preprocessor.transformers[0][2])
    return None


def align_features_to_artifact(fixture_features: pd.DataFrame, artifact) -> pd.DataFrame:
    feature_columns = artifact_feature_columns(artifact)
    if not feature_columns:
        return fixture_features

    aligned = fixture_features.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = 0.0
    return aligned[feature_columns]


def predict_outcome_from_artifact(
    fixture_features: pd.DataFrame,
    model_name: str,
    artifact,
) -> dict[str, float]:
    fixture_features = align_features_to_artifact(fixture_features, artifact)

    if isinstance(artifact, dict) and "pipeline" in artifact:
        probs = artifact["pipeline"].predict_proba(fixture_features)[0]
        classes = artifact["label_encoder"].classes_
    elif isinstance(artifact, dict) and "calibrated_model" in artifact:
        probs = artifact["calibrated_model"].predict_proba(fixture_features)[0]
        classes = artifact["label_encoder"].inverse_transform(artifact["calibrated_model"].classes_)
    else:
        probs = artifact.predict_proba(fixture_features)[0]
        classes = artifact.classes_

    return {label: float(prob) for label, prob in zip(classes, probs)}


def predict_outcome(fixture_features: pd.DataFrame, model_name: str = "Logistic Regression") -> dict[str, float]:
    artifact = load_outcome_artifact(model_name)
    return predict_outcome_from_artifact(fixture_features, model_name, artifact)


def predict_score(fixture_features: pd.DataFrame, max_goals: int = 6) -> tuple[float, float, pd.DataFrame, dict[str, float]]:
    artifact = joblib.load(ARTIFACTS_DIR / "score_model.joblib")
    home_xg = float(artifact["home_model"].predict(fixture_features)[0])
    away_xg = float(artifact["away_model"].predict(fixture_features)[0])
    scorelines = scoreline_probabilities(home_xg, away_xg, max_goals=max_goals)
    score_outcome = outcome_from_scorelines(scorelines)
    return home_xg, away_xg, scorelines, score_outcome


def predict_fixture(
    home_team: str,
    away_team: str,
    neutral: int = 1,
    tournament: str = "FIFA World Cup",
    outcome_model: str = "Logistic Regression + xG",
) -> dict:
    latest = load_latest_team_features()
    X = make_fixture_features(home_team, away_team, latest, neutral=neutral, tournament=tournament)
    outcome_probs = predict_outcome(X, model_name=outcome_model)
    home_xg, away_xg, scorelines, score_outcome = predict_score(X)
    return {
        "home_team": home_team,
        "away_team": away_team,
        "neutral": neutral,
        "tournament": tournament,
        "outcome_model": outcome_model,
        "outcome_model_probabilities": outcome_probs,
        "expected_home_goals": home_xg,
        "expected_away_goals": away_xg,
        "score_model_outcome_probabilities": score_outcome,
        "top_scorelines": scorelines.head(10).to_dict(orient="records"),
    }
