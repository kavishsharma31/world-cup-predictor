from __future__ import annotations

import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR, RESULTS_URL, START_YEAR


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_results(url: str = RESULTS_URL) -> pd.DataFrame:
    """Load international match results.

    Priority:
    1. Local file: data/raw/results.csv
    2. Existing processed file: data/processed/clean_matches.csv
    3. Remote URL from the config

    This lets Codex or you run the project even if you manually download
    results.csv and place it in data/raw/.
    """
    local_path = RAW_DIR / "results.csv"
    if local_path.exists():
        return pd.read_csv(local_path, parse_dates=["date"])

    processed_path = PROCESSED_DIR / "clean_matches.csv"
    if processed_path.exists():
        return pd.read_csv(processed_path, parse_dates=["date"])

    try:
        return pd.read_csv(url, parse_dates=["date"])
    except Exception as exc:
        raise RuntimeError(
            "Could not download the match results CSV. Either connect to the internet "
            "or manually download results.csv and place it at data/raw/results.csv."
        ) from exc


def add_result_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add home/draw/away result labels."""
    df = df.copy()

    def label(row: pd.Series) -> str:
        if row["home_score"] > row["away_score"]:
            return "home_win"
        if row["home_score"] < row["away_score"]:
            return "away_win"
        return "draw"

    df["result"] = df.apply(label, axis=1)
    df["home_points"] = df["result"].map({"home_win": 3, "draw": 1, "away_win": 0})
    df["away_points"] = df["result"].map({"home_win": 0, "draw": 1, "away_win": 3})
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["goal_diff"] = df["home_score"] - df["away_score"]
    return df


def clean_results(df: pd.DataFrame, start_year: int = START_YEAR) -> pd.DataFrame:
    """Clean and restrict the dataset for the first MVP."""
    required = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df[required].copy()
    df = df[df["date"].dt.year >= start_year].copy()
    df = df.dropna(subset=["home_score", "away_score", "home_team", "away_team"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = range(len(df))
    df = add_result_label(df)
    return df


def build_clean_dataset(save: bool = True) -> pd.DataFrame:
    ensure_dirs()
    raw = load_raw_results()
    clean = clean_results(raw)
    if save:
        clean.to_csv(PROCESSED_DIR / "clean_matches.csv", index=False)
    return clean


if __name__ == "__main__":
    data = build_clean_dataset(save=True)
    print(data.head())
    print(f"Saved {len(data):,} matches to data/processed/clean_matches.csv")
