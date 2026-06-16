from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

START_YEAR = 2010
RANDOM_STATE = 42

TRAINING_CUTOFF_DATE = "2026-06-10"

FEATURE_COLUMNS = [
    "elo_diff",
    "neutral",
    "is_world_cup",
    "is_qualifier",
    "is_friendly",
    "home_points_per_game_last_5",
    "away_points_per_game_last_5",
    "points_per_game_diff_last_5",
    "home_goal_diff_avg_last_5",
    "away_goal_diff_avg_last_5",
    "goal_diff_avg_diff_last_5",
    "home_win_rate_last_5",
    "away_win_rate_last_5",
    "win_rate_diff_last_5",
    "home_goals_for_avg_last_5",
    "away_goals_for_avg_last_5",
    "goals_for_avg_diff_last_5",
    "home_goals_against_avg_last_5",
    "away_goals_against_avg_last_5",
    "goals_against_avg_diff_last_5",
    "home_points_per_game_last_10",
    "away_points_per_game_last_10",
    "points_per_game_diff_last_10",
    "home_goal_diff_avg_last_10",
    "away_goal_diff_avg_last_10",
    "goal_diff_avg_diff_last_10",
    "home_win_rate_last_10",
    "away_win_rate_last_10",
    "win_rate_diff_last_10",
]

XG_FEATURE_COLUMNS = [
    "home_xg_for_avg_last_5",
    "away_xg_for_avg_last_5",
    "xg_for_avg_diff_last_5",
    "home_xg_against_avg_last_5",
    "away_xg_against_avg_last_5",
    "xg_against_avg_diff_last_5",
    "home_non_penalty_xg_for_avg_last_5",
    "away_non_penalty_xg_for_avg_last_5",
    "non_penalty_xg_for_avg_diff_last_5",
    "home_non_penalty_xg_against_avg_last_5",
    "away_non_penalty_xg_against_avg_last_5",
    "non_penalty_xg_against_avg_diff_last_5",
]
