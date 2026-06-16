from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import build_all_features
from src.train_outcome_model import train_outcome_model, train_outcome_model_calibrated
from src.train_score_model import train_score_model
from src.train_xgboost_model import train_xgboost_model, train_xgboost_model_calibrated


def main() -> None:
    print("Building features...")
    features, latest = build_all_features(save=True)
    print(f"Feature rows: {len(features):,}")
    print(f"Teams in latest snapshot: {len(latest):,}")

    print("\nTraining baseline outcome model...")
    _, outcome_baseline_metrics = train_outcome_model(use_xg=False)
    print(json.dumps({k: v for k, v in outcome_baseline_metrics.items() if k != "classification_report"}, indent=2))

    print("\nTraining xG-enhanced outcome model...")
    _, outcome_metrics = train_outcome_model(use_xg=True)
    print(json.dumps({k: v for k, v in outcome_metrics.items() if k != "classification_report"}, indent=2))

    print("\nTraining calibrated xG-enhanced outcome model...")
    _, outcome_calibrated_metrics = train_outcome_model_calibrated(use_xg=True)
    print(json.dumps(outcome_calibrated_metrics, indent=2))

    print("\nTraining score model...")
    _, score_metrics = train_score_model()
    print(json.dumps(score_metrics, indent=2))

    print("\nTraining baseline XGBoost outcome model...")
    _, xgboost_baseline_metrics = train_xgboost_model(use_xg=False)
    print(json.dumps(xgboost_baseline_metrics, indent=2))

    print("\nTraining xG-enhanced XGBoost outcome model...")
    _, xgboost_metrics = train_xgboost_model(use_xg=True)
    print(json.dumps(xgboost_metrics, indent=2))

    print("\nTraining calibrated xG-enhanced XGBoost outcome model...")
    _, xgboost_calibrated_metrics = train_xgboost_model_calibrated(use_xg=True)
    print(json.dumps(xgboost_calibrated_metrics, indent=2))

    print("\nDone. Run the app with:")
    print("streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
