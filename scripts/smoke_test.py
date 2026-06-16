from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


failures: list[str] = []
warnings: list[str] = []


def pass_msg(message: str) -> None:
    print(f"PASS: {message}")


def warn_msg(message: str) -> None:
    warnings.append(message)
    print(f"WARNING: {message}")


def fail_msg(message: str) -> None:
    failures.append(message)
    print(f"FAIL: {message}")


def require(condition: bool, message: str) -> bool:
    if condition:
        pass_msg(message)
        return True
    fail_msg(message)
    return False


def check_config_imports() -> None:
    try:
        from src.config import ARTIFACTS_DIR, FEATURE_COLUMNS, PROCESSED_DIR, TRAINING_CUTOFF_DATE
    except Exception as exc:
        fail_msg(f"Config imports failed: {exc}")
        return

    require(ARTIFACTS_DIR.exists(), f"ARTIFACTS_DIR exists: {ARTIFACTS_DIR}")
    require(PROCESSED_DIR.exists(), f"PROCESSED_DIR exists: {PROCESSED_DIR}")
    require(bool(TRAINING_CUTOFF_DATE), f"TRAINING_CUTOFF_DATE is set: {TRAINING_CUTOFF_DATE}")
    require(bool(FEATURE_COLUMNS), f"FEATURE_COLUMNS loaded with {len(FEATURE_COLUMNS)} columns")


def check_fixture_file() -> pd.DataFrame | None:
    from src.config import PROCESSED_DIR

    fixture_path = PROCESSED_DIR / "world_cup_2026_fixtures.csv"
    required_columns = [
        "match_id",
        "date",
        "kickoff_time",
        "group",
        "team_a",
        "team_b",
        "venue_country",
        "venue_city",
        "stage",
    ]

    if not require(fixture_path.exists(), f"Fixture file exists: {fixture_path}"):
        return None

    try:
        fixtures = pd.read_csv(fixture_path)
    except Exception as exc:
        fail_msg(f"Could not load fixture file: {exc}")
        return None

    missing_columns = [column for column in required_columns if column not in fixtures.columns]
    require(not missing_columns, f"Fixture required columns present: {required_columns}")
    if missing_columns:
        fail_msg(f"Fixture file missing columns: {missing_columns}")
        return fixtures

    group_fixtures = fixtures[fixtures["stage"].astype(str).str.lower().eq("group stage")].copy()
    require(len(group_fixtures) == 72, f"Fixture file has 72 group-stage fixtures (found {len(group_fixtures)})")

    expected_groups = list("ABCDEFGHIJKL")
    group_counts = group_fixtures.groupby(group_fixtures["group"].astype(str)).size()
    missing_groups = [group for group in expected_groups if group not in group_counts.index]
    wrong_counts = {group: int(group_counts.get(group, 0)) for group in expected_groups if int(group_counts.get(group, 0)) != 6}

    require(not missing_groups, f"Fixture file includes every group A-L")
    require(not wrong_counts, f"Every group A-L has 6 fixtures")
    if wrong_counts:
        fail_msg(f"Group fixture counts: {wrong_counts}")

    return fixtures


def check_xg_file() -> None:
    from src.config import PROCESSED_DIR

    xg_path = PROCESSED_DIR / "xg_team_match_features.csv"
    if not xg_path.exists():
        warn_msg(f"xG file does not exist: {xg_path}")
        return

    try:
        xg = pd.read_csv(xg_path)
    except Exception as exc:
        warn_msg(f"xG file exists but could not be loaded: {exc}")
        return

    print(f"PASS: xG file loaded with {len(xg):,} rows")
    if xg.empty:
        warn_msg("xG file is empty. This is allowed; models should still run without xG signal.")


def expected_metrics_for_artifact(artifact_name: str) -> str | None:
    mapping = {
        "outcome_model.joblib": "outcome_metrics.json",
        "outcome_model_baseline.joblib": "outcome_baseline_metrics.json",
        "outcome_model_xg.joblib": "outcome_xg_metrics.json",
        "outcome_model_calibrated.joblib": "outcome_calibrated_metrics.json",
        "xgboost_outcome_model.joblib": "xgboost_metrics.json",
        "xgboost_outcome_model_baseline.joblib": "xgboost_baseline_metrics.json",
        "xgboost_outcome_model_xg.joblib": "xgboost_xg_metrics.json",
        "xgboost_outcome_model_calibrated.joblib": "xgboost_calibrated_metrics.json",
        "score_model.joblib": "score_metrics.json",
    }
    return mapping.get(artifact_name)


def check_artifacts() -> tuple[str | None, object | None, object | None]:
    from src.config import ARTIFACTS_DIR
    from src.predict import OUTCOME_MODEL_LABELS

    available_outcome_artifacts = [
        (label, ARTIFACTS_DIR / filename)
        for label, filename in OUTCOME_MODEL_LABELS.items()
        if (ARTIFACTS_DIR / filename).exists()
    ]
    require(bool(available_outcome_artifacts), "At least one outcome model artifact exists")

    score_path = ARTIFACTS_DIR / "score_model.joblib"
    require(score_path.exists(), f"Score model artifact exists: {score_path.name}")

    known_artifacts = [path for _, path in available_outcome_artifacts]
    if score_path.exists():
        known_artifacts.append(score_path)

    for artifact_path in known_artifacts:
        expected_metrics = expected_metrics_for_artifact(artifact_path.name)
        if expected_metrics and not (ARTIFACTS_DIR / expected_metrics).exists():
            warn_msg(f"Missing expected metrics file for {artifact_path.name}: {expected_metrics}")

    if not available_outcome_artifacts or not score_path.exists():
        return None, None, None

    model_label, model_path = available_outcome_artifacts[0]
    try:
        outcome_artifact = joblib.load(model_path)
        pass_msg(f"Loaded outcome model artifact with joblib: {model_label} ({model_path.name})")
    except Exception as exc:
        fail_msg(f"Could not load outcome model artifact {model_path.name}: {exc}")
        outcome_artifact = None

    try:
        score_artifact = joblib.load(score_path)
        pass_msg(f"Loaded score model artifact with joblib: {score_path.name}")
    except Exception as exc:
        fail_msg(f"Could not load score model artifact: {exc}")
        score_artifact = None

    return model_label, outcome_artifact, score_artifact


def pick_prediction_teams(fixtures: pd.DataFrame | None, latest: pd.DataFrame) -> tuple[str, str]:
    teams = set(latest["team"].dropna().astype(str))

    if fixtures is not None and {"team_a", "team_b"}.issubset(fixtures.columns):
        for _, row in fixtures.iterrows():
            team_a = str(row["team_a"])
            team_b = str(row["team_b"])
            if team_a in teams and team_b in teams and team_a != team_b:
                return team_a, team_b

    team_list = sorted(teams)
    if len(team_list) < 2:
        raise ValueError("Need at least two teams in latest_team_features.csv for prediction smoke test.")
    return team_list[0], team_list[1]


def check_prediction(fixtures: pd.DataFrame | None, model_label: str | None, outcome_artifact: object | None) -> None:
    if model_label is None or outcome_artifact is None:
        fail_msg("Skipping prediction sanity check because no outcome model loaded.")
        return

    try:
        from src.predict import (
            load_latest_team_features,
            make_fixture_features,
            predict_outcome_from_artifact,
            predict_score,
        )
    except Exception as exc:
        fail_msg(f"Prediction helper imports failed: {exc}")
        return

    try:
        latest = load_latest_team_features()
        team_a, team_b = pick_prediction_teams(fixtures, latest)
        features = make_fixture_features(team_a, team_b, latest, neutral=1, tournament="FIFA World Cup")
        probabilities = predict_outcome_from_artifact(features, model_label, outcome_artifact)
        probability_sum = sum(probabilities.values())
        require(abs(probability_sum - 1.0) <= 0.01, f"Outcome probabilities sum to approximately 1 ({probability_sum:.4f})")
        print(f"PASS: Prediction sanity check used {team_a} vs {team_b}")
        print(f"PASS: Predicted probabilities: {json.dumps(probabilities, indent=2)}")

        home_goals, away_goals, _, _ = predict_score(features)
        pass_msg(f"Score model predicted goals: {team_a} {home_goals:.2f}, {team_b} {away_goals:.2f}")
    except Exception as exc:
        fail_msg(f"Prediction sanity check failed: {exc}")


def check_app_helper_imports() -> None:
    try:
        from src.bracket import build_round_of_32_matches
        from src.fixtures import load_world_cup_fixtures, validate_group_stage_fixtures
        from src.predict import make_fixture_features, predict_outcome, predict_score
    except Exception as exc:
        fail_msg(f"App helper imports failed: {exc}")
        return

    _ = (
        build_round_of_32_matches,
        load_world_cup_fixtures,
        validate_group_stage_fixtures,
        make_fixture_features,
        predict_outcome,
        predict_score,
    )
    pass_msg("Imported key helper functions used by the Streamlit app")


def main() -> int:
    print("World Cup Predictor Smoke Test")
    print("=" * 34)

    check_config_imports()
    fixtures = check_fixture_file()
    check_xg_file()
    model_label, outcome_artifact, _ = check_artifacts()
    check_prediction(fixtures, model_label, outcome_artifact)
    check_app_helper_imports()

    print("=" * 34)
    print(f"Warnings: {len(warnings)}")
    print(f"Failures: {len(failures)}")

    if failures:
        print("Smoke test FAILED")
        return 1

    print("Smoke test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
