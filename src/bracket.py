from __future__ import annotations

from typing import Any

ROUND_OF_32_SLOTS: list[dict[str, Any]] = [
    {"match": 73, "team_a": "2A", "team_b": "2B"},
    {"match": 74, "team_a": "1E", "third_b": ["A", "B", "C", "D", "F"]},
    {"match": 75, "team_a": "1F", "team_b": "2C"},
    {"match": 76, "team_a": "1C", "team_b": "2F"},
    {"match": 77, "team_a": "1I", "third_b": ["C", "D", "F", "G", "H"]},
    {"match": 78, "team_a": "2E", "team_b": "2I"},
    {"match": 79, "team_a": "1A", "third_b": ["C", "E", "F", "H", "I"]},
    {"match": 80, "team_a": "1L", "third_b": ["E", "H", "I", "J", "K"]},
    {"match": 81, "team_a": "1D", "third_b": ["B", "E", "F", "I", "J"]},
    {"match": 82, "team_a": "1G", "third_b": ["A", "E", "H", "I", "J"]},
    {"match": 83, "team_a": "2K", "team_b": "2L"},
    {"match": 84, "team_a": "1H", "team_b": "2J"},
    {"match": 85, "team_a": "1B", "third_b": ["E", "F", "G", "I", "J"]},
    {"match": 86, "team_a": "1J", "team_b": "2H"},
    {"match": 87, "team_a": "1K", "third_b": ["D", "E", "I", "J", "L"]},
    {"match": 88, "team_a": "2D", "team_b": "2G"},
]

ROUND_OF_16_PATH = [
    (89, 73, 75),
    (90, 74, 77),
    (91, 76, 78),
    (92, 79, 80),
    (93, 83, 84),
    (94, 81, 82),
    (95, 86, 88),
    (96, 85, 87),
]

QUARTER_FINAL_PATH = [
    (97, 89, 90),
    (98, 93, 94),
    (99, 91, 92),
    (100, 95, 96),
]

SEMI_FINAL_PATH = [
    (101, 97, 98),
    (102, 99, 100),
]

FINAL_PATH = (103, 101, 102)


def finisher_code(row: dict[str, int | str]) -> str:
    return f"{int(row['Group Position'])}{row['Group']}"


def rank_third_place_rows(rows: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
    return sorted(
        rows,
        key=lambda row: (int(row["Points"]), int(row["Goal Difference"]), int(row["Goals For"])),
        reverse=True,
    )


def assign_third_place_team(
    allowed_groups: list[str],
    third_place_rows: list[dict[str, int | str]],
    used_teams: set[str],
) -> tuple[dict[str, int | str] | None, bool]:
    for row in third_place_rows:
        if str(row["Team"]) not in used_teams and str(row["Group"]) in allowed_groups:
            used_teams.add(str(row["Team"]))
            return row, False

    # True FIFA Annex C mapping can be added later; this fallback keeps simulations running for now.
    for row in third_place_rows:
        if str(row["Team"]) not in used_teams:
            used_teams.add(str(row["Team"]))
            return row, True

    return None, True


def build_round_of_32_matches(
    qualified_rows: list[dict[str, int | str]],
) -> tuple[list[dict[str, str | int]], bool]:
    direct_slots = {
        finisher_code(row): row
        for row in qualified_rows
        if int(row["Group Position"]) in (1, 2)
    }
    third_place_rows = rank_third_place_rows(
        [row for row in qualified_rows if int(row["Group Position"]) == 3]
    )
    used_third_place_teams: set[str] = set()
    matches = []
    used_fallback = False

    for slot in ROUND_OF_32_SLOTS:
        team_a_row = direct_slots.get(str(slot["team_a"]))
        if team_a_row is None:
            used_fallback = True
            continue

        if "team_b" in slot:
            team_b_row = direct_slots.get(str(slot["team_b"]))
            if team_b_row is None:
                used_fallback = True
                continue
            team_b_slot = str(slot["team_b"])
        else:
            team_b_row, slot_used_fallback = assign_third_place_team(
                slot["third_b"], third_place_rows, used_third_place_teams
            )
            used_fallback = used_fallback or slot_used_fallback
            if team_b_row is None:
                continue
            team_b_slot = finisher_code(team_b_row)

        matches.append(
            {
                "match": int(slot["match"]),
                "team_a": str(team_a_row["Team"]),
                "team_b": str(team_b_row["Team"]),
                "slot_a": str(slot["team_a"]),
                "slot_b": team_b_slot,
            }
        )

    return matches, used_fallback
