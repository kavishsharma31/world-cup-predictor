from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, PROJECT_ROOT

FIXTURES_PATH = PROCESSED_DIR / "world_cup_2026_fixtures.csv"
REQUIRED_FIXTURE_COLUMNS = ["match_id", "date", "group", "team_a", "team_b", "venue_country", "venue_city", "stage"]
FIXTURE_COLUMNS = [
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
EXPECTED_GROUP_TEAMS = 4
EXPECTED_GROUP_FIXTURES = 6


def load_world_cup_fixtures(path: Path = FIXTURES_PATH) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, f"Fixture CSV not found at {path.relative_to(PROJECT_ROOT)}."

    fixtures = pd.read_csv(path, dtype={"match_id": str, "group": str})
    missing_columns = sorted(set(REQUIRED_FIXTURE_COLUMNS) - set(fixtures.columns))
    if missing_columns:
        return None, f"Fixture CSV is missing columns: {', '.join(missing_columns)}."

    if "kickoff_time" not in fixtures.columns:
        fixtures["kickoff_time"] = ""

    fixtures = fixtures[FIXTURE_COLUMNS].copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return fixtures, None


def group_stage_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    return fixtures[fixtures["stage"].astype(str).str.strip().str.lower() == "group stage"].copy()


def validate_group_stage_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    group_fixtures = group_stage_fixtures(fixtures)
    rows = []

    for group, matches in group_fixtures.groupby(group_fixtures["group"].astype(str), sort=True):
        unique_teams = set(matches["team_a"].dropna().astype(str)) | set(matches["team_b"].dropna().astype(str))
        team_count = len(unique_teams)
        fixture_count = len(matches)
        is_complete = team_count == EXPECTED_GROUP_TEAMS and fixture_count == EXPECTED_GROUP_FIXTURES
        rows.append(
            {
                "Group": group,
                "Number of teams": team_count,
                "Number of fixtures": fixture_count,
                "Status": "Complete" if is_complete else "Incomplete",
            }
        )

    return pd.DataFrame(rows, columns=["Group", "Number of teams", "Number of fixtures", "Status"])
