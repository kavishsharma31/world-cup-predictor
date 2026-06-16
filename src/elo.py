from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

BASE_ELO = 1500.0
HOME_ADVANTAGE_ELO = 60.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def actual_score(home_score: int, away_score: int) -> float:
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5


def k_factor(tournament: str) -> float:
    name = str(tournament).lower()
    if "fifa world cup" == name:
        return 60.0
    if "world cup qualification" in name:
        return 40.0
    if "uefa euro" in name or "copa américa" in name or "africa cup" in name or "asian cup" in name:
        return 50.0
    if "friendly" in name:
        return 20.0
    return 30.0


def margin_multiplier(goal_diff: int, elo_diff: float) -> float:
    """Simple margin-of-victory adjustment. Keeps updates stable."""
    margin = abs(goal_diff)
    if margin <= 1:
        return 1.0
    return float(np.log(margin + 1.0) * (2.2 / ((abs(elo_diff) * 0.001) + 2.2)))


def add_elo_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add pre-match and post-match Elo ratings.

    The model should use only `*_elo_before` columns as features.
    Post-match ratings are saved only to build the latest team table.
    """
    df = matches.sort_values(["date", "match_id"]).copy().reset_index(drop=True)
    ratings: dict[str, float] = defaultdict(lambda: BASE_ELO)

    rows = []
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_before = float(ratings[home])
        away_before = float(ratings[away])

        home_adv = 0.0 if int(row["neutral"]) == 1 else HOME_ADVANTAGE_ELO
        expected_home = expected_score(home_before + home_adv, away_before)
        actual_home = actual_score(int(row["home_score"]), int(row["away_score"]))
        elo_diff = (home_before + home_adv) - away_before
        mov = margin_multiplier(int(row["home_score"] - row["away_score"]), elo_diff)
        update = k_factor(row["tournament"]) * mov * (actual_home - expected_home)

        home_after = home_before + update
        away_after = away_before - update
        ratings[home] = home_after
        ratings[away] = away_after

        new_row = row.to_dict()
        new_row["home_elo_before"] = home_before
        new_row["away_elo_before"] = away_before
        new_row["elo_diff"] = home_before - away_before
        new_row["home_elo_after"] = home_after
        new_row["away_elo_after"] = away_after
        rows.append(new_row)

    return pd.DataFrame(rows)
