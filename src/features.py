from __future__ import annotations

import pandas as pd

from src.config import FEATURE_COLUMNS, PROCESSED_DIR, TRAINING_CUTOFF_DATE, XG_FEATURE_COLUMNS
from src.data import build_clean_dataset
from src.elo import add_elo_features

WINDOWS = (5, 10)
XG_FEATURE_PATH = PROCESSED_DIR / "xg_team_match_features.csv"
XG_REQUIRED_COLUMNS = [
    "date",
    "team",
    "opponent",
    "xg_for",
    "xg_against",
    "non_penalty_xg_for",
    "non_penalty_xg_against",
]
XG_SOURCE_COLUMNS = ["xg_for", "xg_against", "non_penalty_xg_for", "non_penalty_xg_against"]
XG_ROLLING_COLUMNS = [f"{column}_avg_last_5" for column in XG_SOURCE_COLUMNS]


def filter_before_training_cutoff(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only matches strictly before the training cutoff date."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out[out["date"] < pd.Timestamp(TRAINING_CUTOFF_DATE)].copy()


def add_tournament_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tournament_lower = df["tournament"].str.lower().fillna("")
    df["is_world_cup"] = (tournament_lower == "fifa world cup").astype(int)
    df["is_qualifier"] = tournament_lower.str.contains("qualification", regex=False).astype(int)
    df["is_friendly"] = tournament_lower.str.contains("friendly", regex=False).astype(int)
    return df


def xg_source_available() -> bool:
    try:
        xg = read_xg_source()
    except pd.errors.ParserError:
        return False
    return xg is not None and not xg.empty


def read_xg_source() -> pd.DataFrame | None:
    if not XG_FEATURE_PATH.exists():
        return None

    try:
        xg = pd.read_csv(XG_FEATURE_PATH)
    except pd.errors.EmptyDataError:
        return None
    missing = sorted(set(XG_REQUIRED_COLUMNS) - set(xg.columns))
    if missing:
        return None
    if xg.empty:
        return None

    xg = xg[XG_REQUIRED_COLUMNS].copy()
    xg["date"] = pd.to_datetime(xg["date"], errors="coerce")
    for column in XG_SOURCE_COLUMNS:
        xg[column] = pd.to_numeric(xg[column], errors="coerce")
    return xg


def load_xg_team_match_features() -> pd.DataFrame | None:
    xg = read_xg_source()
    if xg is None:
        return None

    xg = xg.dropna(subset=["date", "team"])
    xg = xg.dropna(subset=XG_SOURCE_COLUMNS)
    return xg.sort_values(["team", "date"]).reset_index(drop=True)


def build_xg_snapshots(xg: pd.DataFrame) -> pd.DataFrame:
    snapshots = xg[["date", "team", *XG_SOURCE_COLUMNS]].copy()
    for source_column, rolling_column in zip(XG_SOURCE_COLUMNS, XG_ROLLING_COLUMNS):
        snapshots[rolling_column] = (
            snapshots.groupby("team", group_keys=False)[source_column]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return snapshots[["date", "team", *XG_ROLLING_COLUMNS]].sort_values(["date", "team"]).reset_index(drop=True)


def add_side_xg_features(
    matches: pd.DataFrame,
    snapshots: pd.DataFrame,
    team_column: str,
    side_prefix: str,
) -> pd.DataFrame:
    if snapshots.empty:
        out = matches.copy()
        for column in XG_ROLLING_COLUMNS:
            out[f"{side_prefix}_{column}"] = 0.0
        return out

    left = matches[["match_id", "date", team_column]].copy()
    left["date"] = pd.to_datetime(left["date"])
    right = snapshots.rename(columns={"team": team_column})

    merged = pd.merge_asof(
        left.sort_values(["date", team_column]),
        right.sort_values(["date", team_column]),
        on="date",
        by=team_column,
        direction="backward",
        allow_exact_matches=False,
    )
    rename_map = {column: f"{side_prefix}_{column}" for column in XG_ROLLING_COLUMNS}
    merged = merged[["match_id", *XG_ROLLING_COLUMNS]].rename(columns=rename_map)

    out = matches.merge(merged, on="match_id", how="left")
    for column in rename_map.values():
        out[column] = out[column].fillna(0.0)
    return out


def fill_xg_feature_columns(matches: pd.DataFrame) -> pd.DataFrame:
    out = matches.copy()
    for column in XG_FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
        else:
            out[column] = out[column].fillna(0.0)
    return out


def add_xg_features_to_matches(matches: pd.DataFrame, xg: pd.DataFrame | None) -> tuple[pd.DataFrame, bool]:
    if xg is None:
        return matches, False

    snapshots = build_xg_snapshots(xg)
    out = add_side_xg_features(matches, snapshots, "home_team", "home")
    out = add_side_xg_features(out, snapshots, "away_team", "away")

    for source_column in XG_SOURCE_COLUMNS:
        home_column = f"home_{source_column}_avg_last_5"
        away_column = f"away_{source_column}_avg_last_5"
        out[f"{source_column}_avg_diff_last_5"] = out[home_column] - out[away_column]

    return fill_xg_feature_columns(out), True


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    return FEATURE_COLUMNS + [column for column in XG_FEATURE_COLUMNS if column in df.columns]


def xg_features_used(df: pd.DataFrame) -> bool:
    return all(column in df.columns for column in XG_FEATURE_COLUMNS)


def to_team_match_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert match table to one row per team per match."""
    home = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "date": df["date"],
            "team": df["home_team"],
            "opponent": df["away_team"],
            "is_home": 1,
            "goals_for": df["home_score"],
            "goals_against": df["away_score"],
            "points": df["home_points"],
            "elo_before": df["home_elo_before"],
            "elo_after": df["home_elo_after"],
        }
    )
    away = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "date": df["date"],
            "team": df["away_team"],
            "opponent": df["home_team"],
            "is_home": 0,
            "goals_for": df["away_score"],
            "goals_against": df["home_score"],
            "points": df["away_points"],
            "elo_before": df["away_elo_before"],
            "elo_after": df["away_elo_after"],
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    long["win"] = (long["points"] == 3).astype(int)
    long["draw"] = (long["points"] == 1).astype(int)
    long["loss"] = (long["points"] == 0).astype(int)
    long["goal_diff"] = long["goals_for"] - long["goals_against"]
    return long.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def add_rolling_features_to_long(long: pd.DataFrame) -> pd.DataFrame:
    """Add rolling pre-match features.

    Uses shift(1), so the current match result is never used for that match's features.
    """
    long = long.copy()
    metrics = {
        "points": "points_per_game",
        "goals_for": "goals_for_avg",
        "goals_against": "goals_against_avg",
        "goal_diff": "goal_diff_avg",
        "win": "win_rate",
        "draw": "draw_rate",
        "loss": "loss_rate",
    }

    for window in WINDOWS:
        for metric, out_name in metrics.items():
            long[f"{out_name}_last_{window}"] = (
                long.groupby("team", group_keys=False)[metric]
                .apply(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
                .fillna(0.0)
            )

    return long


def build_model_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    df = add_tournament_flags(df)
    long = add_rolling_features_to_long(to_team_match_long(df))

    home_features = long[long["is_home"] == 1].copy()
    away_features = long[long["is_home"] == 0].copy()

    keep_cols = ["match_id"]
    for window in WINDOWS:
        keep_cols += [
            f"points_per_game_last_{window}",
            f"goals_for_avg_last_{window}",
            f"goals_against_avg_last_{window}",
            f"goal_diff_avg_last_{window}",
            f"win_rate_last_{window}",
            f"draw_rate_last_{window}",
            f"loss_rate_last_{window}",
        ]

    home_features = home_features[keep_cols].add_prefix("home_").rename(columns={"home_match_id": "match_id"})
    away_features = away_features[keep_cols].add_prefix("away_").rename(columns={"away_match_id": "match_id"})

    out = df.merge(home_features, on="match_id", how="left").merge(away_features, on="match_id", how="left")
    out, used_xg = add_xg_features_to_matches(out, load_xg_team_match_features())

    for window in WINDOWS:
        out[f"points_per_game_diff_last_{window}"] = (
            out[f"home_points_per_game_last_{window}"] - out[f"away_points_per_game_last_{window}"]
        )
        out[f"goal_diff_avg_diff_last_{window}"] = (
            out[f"home_goal_diff_avg_last_{window}"] - out[f"away_goal_diff_avg_last_{window}"]
        )
        out[f"win_rate_diff_last_{window}"] = (
            out[f"home_win_rate_last_{window}"] - out[f"away_win_rate_last_{window}"]
        )
        out[f"goals_for_avg_diff_last_{window}"] = (
            out[f"home_goals_for_avg_last_{window}"] - out[f"away_goals_for_avg_last_{window}"]
        )
        out[f"goals_against_avg_diff_last_{window}"] = (
            out[f"home_goals_against_avg_last_{window}"] - out[f"away_goals_against_avg_last_{window}"]
        )

    for col in FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0

    return out, long, used_xg


def add_latest_xg_features(latest: pd.DataFrame) -> pd.DataFrame:
    xg = load_xg_team_match_features()
    if xg is None:
        return latest

    xg = xg[xg["date"] < pd.Timestamp(TRAINING_CUTOFF_DATE)].copy()
    if xg.empty:
        latest = latest.copy()
        for column in XG_ROLLING_COLUMNS:
            latest[column] = 0.0
        return latest

    snapshots = build_xg_snapshots(xg).sort_values(["team", "date"])
    latest = latest.copy()
    latest_xg = snapshots.groupby("team", as_index=False).tail(1)
    latest = latest.merge(latest_xg[["team", *XG_ROLLING_COLUMNS]], on="team", how="left")
    for column in XG_ROLLING_COLUMNS:
        latest[column] = latest[column].fillna(0.0)
    return latest


def build_latest_team_features(match_features: pd.DataFrame, long: pd.DataFrame, used_xg: bool = False) -> pd.DataFrame:
    """Build current team features for predicting future fixtures."""
    actual_metrics = ["points", "goals_for", "goals_against", "goal_diff", "win", "draw", "loss"]
    long = long.sort_values(["team", "date", "match_id"]).copy()

    rows = []
    for team, group in long.groupby("team"):
        row: dict[str, float | str] = {"team": team}
        row["current_elo"] = float(group.iloc[-1]["elo_after"])
        row["last_match_date"] = group.iloc[-1]["date"]
        for window in WINDOWS:
            tail = group.tail(window)
            row[f"points_per_game_last_{window}"] = float(tail["points"].mean()) if len(tail) else 0.0
            row[f"goals_for_avg_last_{window}"] = float(tail["goals_for"].mean()) if len(tail) else 0.0
            row[f"goals_against_avg_last_{window}"] = float(tail["goals_against"].mean()) if len(tail) else 0.0
            row[f"goal_diff_avg_last_{window}"] = float(tail["goal_diff"].mean()) if len(tail) else 0.0
            row[f"win_rate_last_{window}"] = float(tail["win"].mean()) if len(tail) else 0.0
            row[f"draw_rate_last_{window}"] = float(tail["draw"].mean()) if len(tail) else 0.0
            row[f"loss_rate_last_{window}"] = float(tail["loss"].mean()) if len(tail) else 0.0
        rows.append(row)

    latest = pd.DataFrame(rows).sort_values("team").reset_index(drop=True)
    if used_xg:
        latest = add_latest_xg_features(latest)
    return latest


def build_all_features(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = filter_before_training_cutoff(build_clean_dataset(save=True))
    with_elo = add_elo_features(clean)
    features, long, used_xg = build_model_feature_table(with_elo)
    latest = build_latest_team_features(features, long, used_xg=used_xg)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        features.to_csv(PROCESSED_DIR / "model_features.csv", index=False)
        latest.to_csv(PROCESSED_DIR / "latest_team_features.csv", index=False)

    return features, latest


def load_model_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "model_features.csv"
    if not path.exists():
        features, _ = build_all_features(save=True)
        return features

    features = filter_before_training_cutoff(pd.read_csv(path, parse_dates=["date"]))
    saved_has_xg = xg_features_used(features)
    if saved_has_xg != xg_source_available():
        features, _ = build_all_features(save=True)
    return features


if __name__ == "__main__":
    features, latest = build_all_features(save=True)
    print(features.head())
    print(f"Saved {len(features):,} model rows")
    print(f"Saved {len(latest):,} team snapshots")
