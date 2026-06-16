from __future__ import annotations

import math
from typing import Any

import pandas as pd


def poisson_pmf(k: int, lam: float) -> float:
    lam = max(float(lam), 0.01)
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def scoreline_probabilities(home_xg: float, away_xg: float, max_goals: int = 6) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob = poisson_pmf(home_goals, home_xg) * poisson_pmf(away_goals, away_xg)
            if home_goals > away_goals:
                outcome = "home_win"
            elif home_goals < away_goals:
                outcome = "away_win"
            else:
                outcome = "draw"
            rows.append(
                {
                    "scoreline": f"{home_goals}-{away_goals}",
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "outcome": outcome,
                    "probability": prob,
                }
            )

    df = pd.DataFrame(rows)
    df["probability"] = df["probability"] / df["probability"].sum()
    return df.sort_values("probability", ascending=False).reset_index(drop=True)


def outcome_from_scorelines(scorelines: pd.DataFrame) -> dict[str, float]:
    grouped = scorelines.groupby("outcome")["probability"].sum().to_dict()
    return {
        "home_win": float(grouped.get("home_win", 0.0)),
        "draw": float(grouped.get("draw", 0.0)),
        "away_win": float(grouped.get("away_win", 0.0)),
    }
