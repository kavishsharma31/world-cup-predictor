from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR, RAW_DIR

EVENTS_DIR = RAW_DIR / "statsbomb" / "events"
OUTPUT_PATH = PROCESSED_DIR / "xg_team_match_features.csv"
OUTPUT_COLUMNS = [
    "date",
    "team",
    "opponent",
    "xg_for",
    "xg_against",
    "non_penalty_xg_for",
    "non_penalty_xg_against",
]


def name_from(value: Any) -> str | None:
    if isinstance(value, dict):
        return value.get("name")
    if isinstance(value, str):
        return value
    return None


def nested_value(data: dict[str, Any], path: list[str]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def load_event_file(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload, {}

    if not isinstance(payload, dict):
        return [], {}

    for key in ("events", "data"):
        events = payload.get(key)
        if isinstance(events, list):
            return events, payload

    return [], payload


def extract_match_date(metadata: dict[str, Any], events: list[dict[str, Any]]) -> str | None:
    date_candidates = [
        metadata.get("match_date"),
        metadata.get("date"),
        nested_value(metadata, ["match", "match_date"]),
        nested_value(metadata, ["match", "date"]),
    ]

    for event in events:
        date_candidates.extend([event.get("match_date"), event.get("date")])

    for value in date_candidates:
        if value:
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.notna(parsed):
                return str(parsed.date())
    return None


def collect_match_teams(metadata: dict[str, Any], events: list[dict[str, Any]]) -> list[str]:
    teams = {
        name_from(metadata.get("home_team")),
        name_from(metadata.get("away_team")),
        name_from(nested_value(metadata, ["match", "home_team"])),
        name_from(nested_value(metadata, ["match", "away_team"])),
    }

    for event in events:
        teams.add(name_from(event.get("team")))

    return sorted(team for team in teams if team)


def event_opponent(event: dict[str, Any]) -> str | None:
    for key in ("opponent", "opposition", "opposing_team"):
        opponent = name_from(event.get(key))
        if opponent:
            return opponent
    return None


def is_shot_event(event: dict[str, Any]) -> bool:
    return name_from(event.get("type")) == "Shot" or isinstance(event.get("shot"), dict)


def shot_xg(event: dict[str, Any]) -> float | None:
    shot = event.get("shot") if isinstance(event.get("shot"), dict) else {}
    value = shot.get("statsbomb_xg", event.get("shot_statsbomb_xg"))
    if value is None:
        return None

    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed)


def is_penalty(event: dict[str, Any]) -> bool:
    shot = event.get("shot") if isinstance(event.get("shot"), dict) else {}
    shot_type = name_from(shot.get("type")) or name_from(event.get("shot_type")) or ""
    return shot_type.lower() == "penalty" or bool(shot.get("penalty"))


def aggregate_events(events: list[dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if not events:
        return []

    match_date = extract_match_date(metadata, events)
    match_teams = collect_match_teams(metadata, events)
    team_totals = {
        team: {"xg_for": 0.0, "non_penalty_xg_for": 0.0}
        for team in match_teams
    }
    explicit_opponents: dict[str, set[str]] = {}
    shots_seen = False

    for event in events:
        team = name_from(event.get("team"))
        if not team:
            continue

        opponent = event_opponent(event)
        if opponent:
            explicit_opponents.setdefault(team, set()).add(opponent)

        if not is_shot_event(event):
            continue

        xg = shot_xg(event)
        if xg is None:
            continue

        shots_seen = True
        team_totals.setdefault(team, {"xg_for": 0.0, "non_penalty_xg_for": 0.0})
        team_totals[team]["xg_for"] += xg
        if not is_penalty(event):
            team_totals[team]["non_penalty_xg_for"] += xg

    if not shots_seen:
        return []

    rows = []
    all_teams = sorted(team_totals)
    for team in all_teams:
        other_teams = [other for other in all_teams if other != team]
        opponent = ""
        if len(other_teams) == 1:
            opponent = other_teams[0]
        elif explicit_opponents.get(team):
            opponent = "; ".join(sorted(explicit_opponents[team]))

        xg_against = sum(team_totals[other]["xg_for"] for other in other_teams)
        non_penalty_xg_against = sum(team_totals[other]["non_penalty_xg_for"] for other in other_teams)

        rows.append(
            {
                "date": match_date or "",
                "team": team,
                "opponent": opponent,
                "xg_for": round(team_totals[team]["xg_for"], 6),
                "xg_against": round(xg_against, 6),
                "non_penalty_xg_for": round(team_totals[team]["non_penalty_xg_for"], 6),
                "non_penalty_xg_against": round(non_penalty_xg_against, 6),
            }
        )

    return rows


def aggregate_event_file(path: Path) -> list[dict[str, Any]]:
    events, metadata = load_event_file(path)
    return aggregate_events(events, metadata)


def build_xg_features() -> pd.DataFrame | None:
    event_files = sorted(EVENTS_DIR.rglob("*.json")) if EVENTS_DIR.exists() else []
    if not event_files:
        print(f"No StatsBomb event files found in {EVENTS_DIR}. Existing xG CSV was left unchanged.")
        return None

    rows: list[dict[str, Any]] = []
    skipped_files = 0
    for path in event_files:
        try:
            rows.extend(aggregate_event_file(path))
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            skipped_files += 1
            print(f"Skipped {path.name}: {exc}")

    if not rows:
        print("No usable xG rows were built from the event files. Existing xG CSV was left unchanged.")
        return None

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = parsed_dates.dt.strftime("%Y-%m-%d").fillna("")
    df = df.sort_values(["date", "team", "opponent"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    missing_dates = int((df["date"] == "").sum())
    print(f"Wrote {len(df):,} team-match xG rows to {OUTPUT_PATH}.")
    print(f"Read {len(event_files):,} event files; skipped {skipped_files:,}.")
    if missing_dates:
        print(f"Warning: {missing_dates:,} rows have no match date and will not be used by the xG feature layer.")

    return df


def main() -> None:
    build_xg_features()


if __name__ == "__main__":
    main()
