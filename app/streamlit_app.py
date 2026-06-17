from __future__ import annotations

import json
import random
import shutil
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bracket import (
    FINAL_PATH,
    QUARTER_FINAL_PATH,
    ROUND_OF_16_PATH,
    SEMI_FINAL_PATH,
    build_round_of_32_matches,
)
from src.config import ARTIFACTS_DIR, PROCESSED_DIR, TRAINING_CUTOFF_DATE
from src.fixtures import (
    EXPECTED_GROUP_FIXTURES,
    FIXTURE_COLUMNS,
    FIXTURES_PATH,
    group_stage_fixtures,
    load_world_cup_fixtures,
    validate_group_stage_fixtures,
)
from src.predict import (
    LATEST_XG_COLUMNS,
    OUTCOME_MODEL_LABELS,
    load_outcome_artifact,
    make_fixture_features,
    predict_outcome,
    predict_outcome_from_artifact,
    predict_score,
)

HOST_COUNTRIES_2026 = ["United States", "Mexico", "Canada"]
NEUTRAL_VENUE = "Other / neutral venue"
OUTCOME_METRICS_FILES = {
    "Logistic Regression Baseline": "outcome_baseline_metrics.json",
    "Logistic Regression + xG": "outcome_xg_metrics.json",
    "Logistic Regression + xG Calibrated": "outcome_calibrated_metrics.json",
    "XGBoost Baseline": "xgboost_baseline_metrics.json",
    "XGBoost + xG": "xgboost_xg_metrics.json",
    "XGBoost + xG Calibrated": "xgboost_calibrated_metrics.json",
}
BASELINE_XG_COMPARISON_FILES = {
    "Logistic Regression Baseline": "outcome_baseline_metrics.json",
    "Logistic Regression + xG": "outcome_xg_metrics.json",
    "XGBoost Baseline": "xgboost_baseline_metrics.json",
    "XGBoost + xG": "xgboost_xg_metrics.json",
}
GROUP_RESULT_KEYS = ["team_a_win", "draw", "team_b_win"]
SIMULATION_OPTIONS = [1000, 5000, 10000]
TEAM_NAME_ALIASES = {
    "Curacao": "Curaçao",
    "Czechia": "Czech Republic",
}
PLACEHOLDER_KICKOFF_VALUES = {"", "tbd", "to be determined", "placeholder", "na", "n/a", "nan", "none", "-"}
PLAYER_AVAILABILITY_PATH = PROCESSED_DIR / "player_availability.csv"
LINEUPS_PATH = PROCESSED_DIR / "lineups.csv"
PLAYER_AVAILABILITY_COLUMNS = ["match_id", "team", "player", "status", "reason", "importance"]
LINEUP_COLUMNS = ["match_id", "team", "player", "position", "is_starter"]
PUBLIC_ERROR_MESSAGE = "Something went wrong while loading this section. Please try again later."


def log_server_error(context: str, error: BaseException) -> None:
    print(f"[world-cup-predictor] {context}: {type(error).__name__}: {error}", file=sys.stderr)


def render_public_error() -> None:
    st.error(PUBLIC_ERROR_MESSAGE)


def safe_render_section(context: str, renderer, *args, **kwargs) -> None:
    try:
        renderer(*args, **kwargs)
    except Exception as exc:
        log_server_error(context, exc)
        render_public_error()


def safe_load_world_cup_fixtures(context: str) -> tuple[pd.DataFrame | None, str | None]:
    try:
        fixtures, warning = load_world_cup_fixtures()
    except Exception as exc:
        log_server_error(context, exc)
        return None, PUBLIC_ERROR_MESSAGE

    if warning:
        log_server_error(context, RuntimeError(warning))
        return None, PUBLIC_ERROR_MESSAGE

    return fixtures, None


def load_startup_snapshot() -> pd.DataFrame:
    required_files = [
        PROCESSED_DIR / "latest_team_features.csv",
        ARTIFACTS_DIR / "outcome_model.joblib",
        ARTIFACTS_DIR / "score_model.joblib",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        log_server_error("startup model loading", FileNotFoundError(", ".join(missing)))
        render_public_error()
        st.stop()

    try:
        return pd.read_csv(PROCESSED_DIR / "latest_team_features.csv", parse_dates=["last_match_date"])
    except Exception as exc:
        log_server_error("latest team snapshot loading", exc)
        render_public_error()
        st.stop()


def inject_dashboard_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #070a12;
            --surface-1: #0f1622;
            --surface-2: #151f2e;
            --surface-3: #1b293b;
            --surface-note: #111a27;
            --border: #2b394c;
            --border-strong: #46566d;
            --text: #f3f6fa;
            --muted: #a8b3c3;
            --subtle: #7d8999;
            --accent: #49b47e;
            --accent-strong: #72d19c;
            --rule: #223044;
            --radius: 8px;
        }
        .stApp {
            background: var(--page-bg);
            color: var(--text);
        }
        #MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbar"], [data-testid="stStatusWidget"] {
            visibility: hidden;
        }
        [data-testid="stHeader"] {
            background: var(--page-bg);
        }
        .block-container {
            max-width: 1440px;
            padding-top: 1.25rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3, h4, h5, h6, p, li, label, span {
            color: var(--text);
        }
        p, li, label {
            font-size: 0.95rem;
            line-height: 1.55;
        }
        h2, h3 {
            letter-spacing: 0;
        }
        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border-strong);
            background: var(--surface-1);
            border-radius: var(--radius);
            padding: 28px 30px;
            margin-bottom: 24px;
        }
        .hero-topline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            border-bottom: 1px solid var(--rule);
            margin-bottom: 22px;
            padding-bottom: 14px;
        }
        .hero-kicker {
            color: var(--accent-strong);
            font-size: 0.75rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .hero-live {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .hero h1 {
            margin: 0 0 12px;
            font-size: 3rem;
            line-height: 1.05;
            letter-spacing: 0;
            font-weight: 850;
        }
        .hero p {
            margin: 0;
            max-width: 920px;
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.6;
        }
        .hero-stats {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin-top: 24px;
        }
        .hero-stat {
            background: var(--surface-2);
            border: 1px solid var(--border-strong);
            border-radius: var(--radius);
            padding: 20px;
        }
        .hero-stat span {
            display: block;
            color: var(--subtle);
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .hero-stat strong {
            display: block;
            margin-top: 8px;
            color: var(--text);
            font-size: 1.08rem;
            font-weight: 800;
        }
        .section-card, div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface-1);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.15rem;
        }
        div[data-testid="stMetric"], .kpi-card, .prob-card, .score-card, .method-card {
            background: var(--surface-2);
            border: 1px solid var(--border-strong);
            border-radius: var(--radius);
            padding: 18px;
        }
        div[data-testid="stMetric"] label, div[data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
        }
        div[data-testid="stMetricValue"] {
            color: var(--text);
        }
        .prob-card {
            min-height: 166px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            border-left: 4px solid var(--border-strong);
            background: var(--surface-2);
            position: relative;
            overflow: hidden;
        }
        .prob-card.top {
            border-color: var(--accent);
            border-left-color: var(--accent);
            background: #162433;
        }
        .prob-card .label, .score-card .label, .kpi-card .label {
            color: var(--subtle);
            font-size: 0.75rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .prob-card .value {
            color: var(--text);
            font-size: 2.75rem;
            font-weight: 800;
            line-height: 1;
        }
        .prob-card .tag {
            color: var(--accent-strong);
            font-size: 0.82rem;
            font-weight: 700;
        }
        .prob-card .bar {
            height: 7px;
            border-radius: 999px;
            background: #253246;
            overflow: hidden;
        }
        .prob-card .bar span {
            display: block;
            height: 100%;
            border-radius: 999px;
            background: var(--accent);
        }
        .score-card .value, .kpi-card .value {
            color: var(--text);
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.1;
            margin-top: 8px;
        }
        .kpi-card .sub, .score-card .sub, .method-card p {
            color: var(--muted);
            margin: 6px 0 0;
            font-size: 0.9rem;
        }
        .section-heading {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 16px;
            border-bottom: 1px solid var(--rule);
            margin-bottom: 16px;
            padding-bottom: 12px;
        }
        .section-eyebrow {
            color: var(--accent-strong);
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .section-title {
            color: var(--text);
            font-size: 1.35rem;
            font-weight: 800;
            margin-top: 2px;
        }
        .section-subtitle {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 3px;
            max-width: 820px;
        }
        .info-note {
            border-left: 4px solid var(--accent);
            background: var(--surface-note);
            border-radius: 4px;
            color: var(--muted);
            padding: 11px 14px;
            margin: 10px 0 14px;
            font-size: 0.92rem;
        }
        .bullet-panel {
            border-left: 4px solid var(--accent);
            background: var(--surface-note);
            border-radius: 4px;
            padding: 13px 16px;
            margin-top: 12px;
        }
        .bullet-panel ul {
            margin: 0;
            padding-left: 1.05rem;
        }
        .bullet-panel li {
            color: var(--muted);
            margin: 6px 0;
        }
        .dark-table-wrap {
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow-x: auto;
            overflow-y: hidden;
            background: var(--surface-1);
            margin: 8px 0 16px;
        }
        .dark-table-wrap table {
            width: 100%;
            border-collapse: collapse;
            color: var(--text);
            font-size: 0.88rem;
        }
        .dark-table-wrap thead th {
            background: var(--surface-3);
            color: var(--text);
            font-weight: 800;
            text-transform: uppercase;
            font-size: 0.72rem;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
            text-align: left;
            white-space: nowrap;
        }
        .dark-table-wrap tbody th,
        .dark-table-wrap tbody td {
            padding: 9px 12px;
            border-bottom: 1px solid #202c3e;
            color: var(--muted);
            white-space: nowrap;
        }
        .dark-table-wrap tbody tr:nth-child(even) {
            background: #0c1320;
        }
        .dark-table-wrap tbody tr:hover {
            background: #162234;
        }
        .method-card {
            min-height: 170px;
            margin-bottom: 14px;
        }
        .method-card h4 {
            margin: 0 0 8px;
            color: var(--text);
            font-size: 1.08rem;
            font-weight: 800;
        }
        .method-card ul {
            margin: 0;
            padding-left: 1rem;
        }
        .method-card li {
            color: var(--muted);
            margin-bottom: 4px;
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
            background: var(--surface-1);
        }
        [data-baseweb="tab-list"] {
            gap: 10px;
            margin-bottom: 16px;
        }
        [data-baseweb="tab"] {
            background: var(--surface-1);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 8px 18px;
        }
        [data-baseweb="tab"] p {
            color: var(--muted);
            font-weight: 700;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            border-color: var(--accent);
            background: #1d3b2f;
        }
        [data-baseweb="tab"][aria-selected="true"] p {
            color: var(--text);
            font-weight: 800;
        }
        [data-baseweb="tab-highlight"] {
            background: transparent;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 999px;
            border: 1px solid var(--border-strong);
            background: var(--surface-3);
            color: var(--text);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: var(--accent);
            color: var(--accent-strong);
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            background: var(--surface-3);
            border-color: var(--border-strong);
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="input"] input {
            color: var(--text) !important;
        }
        div[data-baseweb="select"] svg {
            fill: var(--muted);
        }
        .stAlert {
            border-radius: var(--radius);
        }
        @media (max-width: 760px) {
            .hero h1 {
                font-size: 2.25rem;
            }
            .hero-topline {
                align-items: flex-start;
                flex-direction: column;
            }
            .hero-stats {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-topline">
                <div class="hero-kicker">World Cup 2026 analytics dashboard</div>
                <div class="hero-live">Model trained before {escape(TRAINING_CUTOFF_DATE)}</div>
            </div>
            <h1>World Cup Match &amp; Tournament Predictor</h1>
            <p>
                Predict match outcomes, scorelines, and group qualification chances using historical football data,
                Elo, form, xG, and Monte Carlo simulation.
            </p>
            <div class="hero-stats">
                <div class="hero-stat"><span>Predictor</span><strong>Match odds + scores</strong></div>
                <div class="hero-stat"><span>Simulation</span><strong>Group paths</strong></div>
                <div class="hero-stat"><span>Models</span><strong>Baseline + xG</strong></div>
                <div class="hero-stat"><span>Exports</span><strong>CSV ready</strong></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eyebrow_html = f'<div class="section-eyebrow">{escape(eyebrow)}</div>' if eyebrow else ""
    subtitle_html = f'<div class="section-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-heading">
            <div>
                {eyebrow_html}
                <div class="section-title">{escape(title)}</div>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_note(text: str) -> None:
    st.markdown(f'<div class="info-note">{escape(text)}</div>', unsafe_allow_html=True)


def render_probability_card(label: str, probability: float, is_top: bool = False) -> None:
    top_class = " top" if is_top else ""
    tag = "Highest probability" if is_top else "Model probability"
    bar_width = max(0.0, min(100.0, probability * 100))
    st.markdown(
        f"""
        <div class="prob-card{top_class}">
            <div class="label">{escape(label)}</div>
            <div class="value">{probability * 100:.1f}%</div>
            <div class="tag">{tag}</div>
            <div class="bar"><span style="width: {bar_width:.1f}%"></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_card(label: str, value: str, subtext: str) -> None:
    st.markdown(
        f"""
        <div class="score-card">
            <div class="label">{escape(label)}</div>
            <div class="value">{escape(value)}</div>
            <div class="sub">{escape(subtext)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, subtext: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="label">{escape(label)}</div>
            <div class="value">{escape(value)}</div>
            <div class="sub">{escape(subtext)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_method_card(title: str, items: list[str]) -> None:
    body = "".join(f"<li>{escape(item)}</li>" for item in items)
    st.markdown(
        f"""
        <div class="method-card">
            <h4>{escape(title)}</h4>
            <ul>{body}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_bullet_panel(points: list[str]) -> None:
    body = "".join(f"<li>{escape(point)}</li>" for point in points)
    st.markdown(f'<div class="bullet-panel"><ul>{body}</ul></div>', unsafe_allow_html=True)


def render_dark_table(data: pd.DataFrame, hide_index: bool = False) -> None:
    table_html = data.to_html(index=not hide_index, escape=True, border=0)
    st.markdown(f'<div class="dark-table-wrap">{table_html}</div>', unsafe_allow_html=True)


def get_host_side(team_a: str, team_b: str, venue_country: str) -> str | None:
    if venue_country not in HOST_COUNTRIES_2026:
        return None
    if team_a == venue_country:
        return "team_a"
    if team_b == venue_country:
        return "team_b"
    return None


def model_outcome_label(outcome: str, team_a_is_model_home: bool, team_a: str, team_b: str) -> str:
    if outcome == "draw":
        return "Draw"
    if (outcome == "home_win") == team_a_is_model_home:
        return f"{team_a} win"
    return f"{team_b} win"


def team_result_probabilities(model_probs: dict[str, float], team_a_is_model_home: bool) -> dict[str, float]:
    if team_a_is_model_home:
        team_a_win = model_probs.get("home_win", 0.0)
        team_b_win = model_probs.get("away_win", 0.0)
    else:
        team_a_win = model_probs.get("away_win", 0.0)
        team_b_win = model_probs.get("home_win", 0.0)
    return {
        "team_a_win": team_a_win,
        "draw": model_probs.get("draw", 0.0),
        "team_b_win": team_b_win,
    }


def host_status_text(host_side: str | None, team_a: str, team_b: str, venue_country: str) -> str:
    if host_side == "team_a":
        return f"Host advantage: Team A ({team_a}) in {venue_country}"
    if host_side == "team_b":
        return f"Host advantage: Team B ({team_b}) in {venue_country}"
    return "Neutral venue: no host advantage"


def neutral_status_text(host_side: str | None) -> str:
    return "Neutral venue" if host_side is None else "Not neutral"


def host_advantage_status_text(host_side: str | None, team_a: str, team_b: str) -> str:
    if host_side == "team_a":
        return f"Team A ({team_a})"
    if host_side == "team_b":
        return f"Team B ({team_b})"
    return "None"


def model_mapping_for_match(team_a: str, team_b: str, venue_country: str) -> tuple[str, str, bool, int, str | None]:
    host_side = get_host_side(team_a, team_b, venue_country)
    neutral = 0 if host_side else 1
    if host_side == "team_b":
        return team_b, team_a, False, neutral, host_side
    return team_a, team_b, True, neutral, host_side


def model_team_name(team: str) -> str:
    return TEAM_NAME_ALIASES.get(team, team)


def build_explanation_points(
    team_a: str,
    team_b: str,
    outcome_probs: dict[str, float],
    elo_diff: float,
    ppg_diff: float,
    goal_diff_diff: float,
    win_rate_diff: float,
    neutral_status: str,
    host_advantage_status: str,
) -> list[str]:
    best_key = max(outcome_probs, key=outcome_probs.get)
    best_label = {"team_a_win": f"{team_a} win", "draw": "draw", "team_b_win": f"{team_b} win"}[best_key]
    points = [f"The outcome model puts the highest probability on a {best_label}, but this is an estimate, not a certainty."]

    if abs(elo_diff) >= 50:
        stronger_team = team_a if elo_diff > 0 else team_b
        points.append(f"{stronger_team} has the higher Elo rating by about {abs(elo_diff):.0f} points.")
    else:
        points.append("The teams are close on Elo, so rating difference alone does not strongly separate them.")

    if abs(ppg_diff) >= 0.25 or abs(goal_diff_diff) >= 0.25:
        form_team = team_a if (ppg_diff + goal_diff_diff) > 0 else team_b
        points.append(f"Recent last-5 form leans toward {form_team} across points per game and goal difference.")
    else:
        points.append("Recent last-5 points per game and goal difference are fairly close.")

    if abs(win_rate_diff) >= 0.15:
        win_rate_team = team_a if win_rate_diff > 0 else team_b
        points.append(f"{win_rate_team} has the higher win rate over the last 5 matches.")

    if host_advantage_status == "None":
        points.append(f"Venue input: {neutral_status}, so no host advantage is applied.")
    else:
        points.append(f"Venue input: {neutral_status}; host advantage is applied to {host_advantage_status}.")
    return points[:5]


def read_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_number(value: object, digits: int = 3) -> str:
    if value is None:
        return "Not available"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_percent(value: object) -> str:
    if value is None:
        return "Not available"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def available_outcome_models() -> list[str]:
    return [label for label, filename in OUTCOME_MODEL_LABELS.items() if (ARTIFACTS_DIR / filename).exists()]


def feature_columns_from_artifact(artifact) -> list[str] | None:
    if isinstance(artifact, dict):
        return artifact.get("feature_columns")

    feature_columns = getattr(artifact, "feature_columns_", None)
    if feature_columns is not None:
        return list(feature_columns)

    if hasattr(artifact, "named_steps"):
        preprocessor = artifact.named_steps.get("preprocess")
        if preprocessor is not None and getattr(preprocessor, "transformers", None):
            return list(preprocessor.transformers[0][2])

    return None


def fitted_pipeline_from_artifact(artifact):
    if isinstance(artifact, dict):
        if "pipeline" in artifact:
            return artifact["pipeline"]
        if "calibrated_model" in artifact:
            return getattr(artifact["calibrated_model"], "estimator", None)

    if hasattr(artifact, "estimator"):
        return getattr(artifact, "estimator", None)

    return artifact


def feature_importance_table(model_label: str, artifact, top_n: int = 15) -> pd.DataFrame | None:
    feature_columns = feature_columns_from_artifact(artifact)
    pipeline = fitted_pipeline_from_artifact(artifact)
    if not feature_columns or pipeline is None or not hasattr(pipeline, "named_steps"):
        return None

    model = pipeline.named_steps.get("model")
    if model is None:
        return None

    if hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_)
        if coefficients.ndim == 1:
            importances = np.abs(coefficients)
        else:
            importances = np.mean(np.abs(coefficients), axis=0)
    elif hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    else:
        return None

    if len(importances) != len(feature_columns):
        return None

    table = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": importances.astype(float),
            "Model": model_label,
        }
    )
    return table.sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)


def display_feature_name(feature: str, team_a_is_model_home: bool) -> str:
    home_label = "Team A" if team_a_is_model_home else "Team B"
    away_label = "Team B" if team_a_is_model_home else "Team A"

    if feature == "elo_diff":
        return f"Elo difference ({home_label} - {away_label})"
    if feature.endswith("_diff_last_5") or feature.endswith("_diff_last_10"):
        readable = feature.replace("_diff_", f" ({home_label} - {away_label}) ").replace("_", " ")
        return readable.title()
    if feature.startswith("home_"):
        return f"{home_label} {feature.removeprefix('home_').replace('_', ' ')}"
    if feature.startswith("away_"):
        return f"{away_label} {feature.removeprefix('away_').replace('_', ' ')}"
    return feature.replace("_", " ").title()


def local_logistic_contribution_table(
    model_label: str,
    artifact,
    fixture_features: pd.DataFrame,
    outcome_probs: dict[str, float],
    team_a_is_model_home: bool,
    top_n: int = 10,
) -> pd.DataFrame | None:
    feature_columns = feature_columns_from_artifact(artifact)
    pipeline = fitted_pipeline_from_artifact(artifact)
    if not feature_columns or pipeline is None or not hasattr(pipeline, "named_steps"):
        return None

    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocess")
    if model is None or preprocessor is None or not hasattr(model, "coef_"):
        return None

    aligned_features = fixture_features.copy()
    for column in feature_columns:
        if column not in aligned_features.columns:
            aligned_features[column] = 0.0
    aligned_features = aligned_features[feature_columns]

    transformed_values = preprocessor.transform(aligned_features)
    if hasattr(transformed_values, "toarray"):
        transformed_values = transformed_values.toarray()
    values = np.asarray(transformed_values)[0]

    coefficients = np.asarray(model.coef_)
    classes = list(getattr(model, "classes_", []))
    predicted_class = max(outcome_probs, key=outcome_probs.get)
    if predicted_class not in classes:
        return None

    if coefficients.ndim == 1:
        selected_coefficients = coefficients
    elif coefficients.shape[0] == 1 and len(classes) == 2:
        selected_coefficients = coefficients[0] if predicted_class == classes[1] else -coefficients[0]
    else:
        selected_coefficients = coefficients[classes.index(predicted_class)]

    if len(selected_coefficients) != len(feature_columns):
        return None

    contributions = values * selected_coefficients
    table = pd.DataFrame(
        {
            "Feature": [display_feature_name(feature, team_a_is_model_home) for feature in feature_columns],
            "Value": values.astype(float),
            "Contribution": contributions.astype(float),
            "Direction": [
                "Supports predicted outcome" if contribution >= 0 else "Works against predicted outcome"
                for contribution in contributions
            ],
            "abs_contribution": np.abs(contributions),
        }
    )
    positive = table[table["Contribution"] >= 0].sort_values("abs_contribution", ascending=False).head(top_n // 2)
    negative = table[table["Contribution"] < 0].sort_values("abs_contribution", ascending=False).head(top_n // 2)
    selected = pd.concat([positive, negative])

    if len(selected) < top_n:
        remaining = table.drop(selected.index, errors="ignore").sort_values("abs_contribution", ascending=False)
        selected = pd.concat([selected, remaining.head(top_n - len(selected))])

    return selected.sort_values("abs_contribution", ascending=False).head(top_n).drop(columns="abs_contribution")


def render_csv_download(df: pd.DataFrame, file_name: str, label: str, key: str) -> None:
    if df is None or df.empty:
        return
    file_path = Path(file_name)
    if file_path.name != file_name or file_path.suffix.lower() != ".csv":
        log_server_error("blocked unsafe download", ValueError(file_name))
        return
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        key=key,
    )


def placeholder_kickoff_mask(fixtures: pd.DataFrame) -> pd.Series:
    kickoff = fixtures["kickoff_time"].fillna("").astype(str).str.strip().str.lower()
    return kickoff.isin(PLACEHOLDER_KICKOFF_VALUES)


def fixture_quality_checks(fixtures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    group_fixtures = group_stage_fixtures(fixtures)
    fixtures_per_group = (
        group_fixtures.groupby(group_fixtures["group"].astype(str), sort=True)
        .size()
        .reset_index(name="Fixtures")
        .rename(columns={"group": "Group"})
    )
    tbd_venue_count = int(fixtures["venue_city"].fillna("").astype(str).str.strip().str.lower().eq("tbd").sum())
    placeholder_kickoff_count = int(placeholder_kickoff_mask(fixtures).sum())

    summary = pd.DataFrame(
        [
            {"Check": "Total fixtures", "Value": len(fixtures)},
            {"Check": "Group-stage fixtures", "Value": len(group_fixtures)},
            {"Check": "Number of groups", "Value": group_fixtures["group"].nunique()},
            {"Check": "TBD venue_city values", "Value": tbd_venue_count},
            {"Check": "Blank or placeholder kickoff_time values", "Value": placeholder_kickoff_count},
        ]
    )

    warnings = []
    if len(group_fixtures) != 72:
        warnings.append(f"Expected 72 group-stage fixtures, found {len(group_fixtures)}.")
    incomplete_groups = fixtures_per_group[fixtures_per_group["Fixtures"] != EXPECTED_GROUP_FIXTURES]
    if not incomplete_groups.empty:
        group_list = ", ".join(f"{row['Group']} ({row['Fixtures']})" for _, row in incomplete_groups.iterrows())
        warnings.append(f"Expected 6 fixtures per group. Check: {group_list}.")
    if tbd_venue_count:
        warnings.append(f"`venue_city` contains {tbd_venue_count} TBD value(s).")
    if placeholder_kickoff_count:
        warnings.append(f"`kickoff_time` contains {placeholder_kickoff_count} blank or placeholder value(s).")

    return summary, fixtures_per_group, warnings


def validate_uploaded_fixtures(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    try:
        uploaded = pd.read_csv(uploaded_file, dtype={"match_id": str, "group": str})
    except Exception as exc:
        log_server_error("uploaded fixture CSV loading", exc)
        return None, "Uploaded CSV could not be read."

    missing_columns = [column for column in FIXTURE_COLUMNS if column not in uploaded.columns]
    if missing_columns:
        return None, f"Uploaded CSV is missing required columns: {', '.join(missing_columns)}."

    uploaded = uploaded[FIXTURE_COLUMNS].copy()
    uploaded["date"] = pd.to_datetime(uploaded["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if uploaded["date"].isna().any():
        return None, "Uploaded CSV contains invalid date values."

    if group_stage_fixtures(uploaded).empty:
        return None, "Uploaded CSV does not contain any Group Stage fixtures."

    return uploaded, None


def load_optional_context_csv(path: Path, required_columns: list[str]) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return pd.DataFrame(columns=required_columns), None

    try:
        data = pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=required_columns), None
    except Exception as exc:
        log_server_error(f"optional context loading: {path.name}", exc)
        return pd.DataFrame(columns=required_columns), PUBLIC_ERROR_MESSAGE

    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        log_server_error(
            f"optional context validation: {path.name}",
            ValueError(f"missing columns: {', '.join(missing_columns)}"),
        )
        return (
            pd.DataFrame(columns=required_columns),
            PUBLIC_ERROR_MESSAGE,
        )

    return data[required_columns].copy(), None


def rows_for_match_team(data: pd.DataFrame, match_id: str, team: str) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    team_names = {team, model_team_name(team)}
    return data[
        data["match_id"].astype(str).str.strip().eq(str(match_id).strip())
        & data["team"].astype(str).str.strip().isin(team_names)
    ].copy()


def format_lineup_rows(lineups: pd.DataFrame) -> pd.DataFrame:
    if lineups.empty:
        return lineups

    out = lineups[["player", "position", "is_starter"]].copy()
    starter_mask = out["is_starter"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "starter"})
    out["Starter"] = starter_mask.map({True: "Yes", False: "No"})
    out["starter_sort"] = starter_mask.astype(int)
    out = out.sort_values(["starter_sort", "position", "player"], ascending=[False, True, True])
    return out.rename(columns={"player": "Player", "position": "Position"})[["Player", "Position", "Starter"]]


def availability_impact_counts(availability_rows: pd.DataFrame) -> dict[str, int]:
    if availability_rows.empty:
        return {"unavailable": 0, "doubtful": 0}

    importance = availability_rows["importance"].astype(str).str.strip().str.lower()
    status = availability_rows["status"].astype(str).str.strip().str.lower()
    high_importance = importance.isin({"high", "important", "key", "major", "critical"})
    unavailable = status.isin({"unavailable", "out", "injured", "suspended"})
    doubtful = status.isin({"doubtful", "questionable", "uncertain"})

    return {
        "unavailable": int((high_importance & unavailable).sum()),
        "doubtful": int((high_importance & doubtful).sum()),
    }


def render_team_news(match_id: str, team_a: str, team_b: str) -> None:
    availability, availability_warning = load_optional_context_csv(
        PLAYER_AVAILABILITY_PATH, PLAYER_AVAILABILITY_COLUMNS
    )
    lineups, lineups_warning = load_optional_context_csv(LINEUPS_PATH, LINEUP_COLUMNS)

    availability_a = rows_for_match_team(availability, match_id, team_a)
    availability_b = rows_for_match_team(availability, match_id, team_b)
    lineups_a = rows_for_match_team(lineups, match_id, team_a)
    lineups_b = rows_for_match_team(lineups, match_id, team_b)

    with st.container(border=True):
        render_section_header(
            "Team News",
            "Lineups and availability are context only and are not currently included in the model.",
            "Context",
        )

        for warning in [availability_warning, lineups_warning]:
            if warning:
                st.warning(warning)

        if availability_a.empty and availability_b.empty and lineups_a.empty and lineups_b.empty:
            st.info("No lineup or player availability data added for this match yet.")
            return

        impact_a = availability_impact_counts(availability_a)
        impact_b = availability_impact_counts(availability_b)
        render_section_header("Availability Impact", "High-importance unavailable and doubtful players.", "Team news")
        impact_table = pd.DataFrame(
            [
                {
                    "Team": f"Team A: {team_a}",
                    "High-importance unavailable": impact_a["unavailable"],
                    "High-importance doubtful": impact_a["doubtful"],
                },
                {
                    "Team": f"Team B: {team_b}",
                    "High-importance unavailable": impact_b["unavailable"],
                    "High-importance doubtful": impact_b["doubtful"],
                },
            ]
        )
        render_dark_table(impact_table.set_index("Team"))

        warning_messages = []
        for label, team, impact in [("Team A", team_a, impact_a), ("Team B", team_b, impact_b)]:
            if impact["unavailable"] or impact["doubtful"]:
                warning_messages.append(
                    f"{label} ({team}) has {impact['unavailable']} high-importance unavailable "
                    f"and {impact['doubtful']} high-importance doubtful player(s)."
                )
        for message in warning_messages:
            st.warning(message)

        team_columns = st.columns(2)
        for column, label, team, availability_rows, lineup_rows in [
            (team_columns[0], "Team A", team_a, availability_a, lineups_a),
            (team_columns[1], "Team B", team_b, availability_b, lineups_b),
        ]:
            with column:
                render_section_header(f"{label}: {team}", "Local notes for this fixture.", "Squad")

                st.caption("Player availability")
                if availability_rows.empty:
                    st.caption("No availability notes.")
                else:
                    availability_display = availability_rows[["player", "status", "reason", "importance"]].rename(
                        columns={
                            "player": "Player",
                            "status": "Status",
                            "reason": "Reason",
                            "importance": "Importance",
                        }
                    )
                    render_dark_table(availability_display, hide_index=True)

                st.caption("Lineup")
                lineup_display = format_lineup_rows(lineup_rows)
                if lineup_display.empty:
                    st.caption("No lineup added.")
                else:
                    render_dark_table(lineup_display, hide_index=True)


def format_fixture_label(row: pd.Series) -> str:
    group = "" if pd.isna(row["group"]) else str(row["group"])
    stage_group = f"{row['stage']}/Group {group}" if group else str(row["stage"])
    return (
        f"{row['date']} | {stage_group} | {row['team_a']} vs {row['team_b']} | "
        f"{row['venue_city']}, {row['venue_country']}"
    )


def render_predictor(latest: pd.DataFrame, teams: list[str]) -> None:
    latest_by_team = latest.set_index("team")
    selected_match_id: str | None = None

    fixtures, fixture_warning = safe_load_world_cup_fixtures("predictor fixture loading")
    has_fixtures = fixtures is not None and not fixtures.empty
    mode_options = ["Select official fixture", "Custom match"]
    default_mode_index = 0 if has_fixtures else 1

    with st.container(border=True):
        render_section_header(
            "Match Setup",
            "Pick an official fixture or build a custom neutral-site World Cup matchup.",
            "Predictor",
        )
        mode_col, model_col = st.columns(2)
        with mode_col:
            prediction_mode = st.selectbox("Prediction mode", mode_options, index=default_mode_index)
        with model_col:
            outcome_model = st.selectbox("Outcome model", available_outcome_models())

        if fixture_warning:
            st.warning(fixture_warning)

        if prediction_mode == "Select official fixture" and not has_fixtures:
            st.warning("Official fixture mode is unavailable until the fixture CSV is added.")
            prediction_mode = "Custom match"

        if prediction_mode == "Select official fixture":
            fixture_labels = [format_fixture_label(row) for _, row in fixtures.iterrows()]
            selected_label = st.selectbox("Fixture", fixture_labels, index=None, placeholder="Select a fixture")
            if selected_label is None:
                render_info_note(
                    "Team A and Team B are labels only. Host advantage applies only for a selected 2026 host country."
                )
                render_info_note("Select a fixture to generate a prediction.")
                return

            fixture = fixtures.iloc[fixture_labels.index(selected_label)]
            team_a = str(fixture["team_a"])
            team_b = str(fixture["team_b"])
            venue_country = str(fixture["venue_country"])
            selected_match_id = str(fixture["match_id"])
            tournament = "FIFA World Cup"

            fixture_cols = st.columns(4)
            with fixture_cols[0]:
                render_kpi_card("Team A", team_a, "Fixture label")
            with fixture_cols[1]:
                render_kpi_card("Team B", team_b, "Fixture label")
            with fixture_cols[2]:
                render_kpi_card("Venue country", venue_country, "Host logic input")
            with fixture_cols[3]:
                render_kpi_card("Venue city", str(fixture["venue_city"]), "Fixture CSV")
        else:
            selected_match_id = None
            col1, col2, col3 = st.columns(3)
            with col1:
                default_team_a = teams.index("Argentina") if "Argentina" in teams else 0
                team_a = st.selectbox("Team A", teams, index=default_team_a)
            with col2:
                default_team_b = teams.index("France") if "France" in teams else min(1, len(teams) - 1)
                team_b = st.selectbox("Team B", teams, index=default_team_b)
            with col3:
                tournament = st.selectbox("Tournament type", ["FIFA World Cup", "Friendly", "World Cup qualification", "Other"])
                venue_country = st.selectbox("Venue country", [*HOST_COUNTRIES_2026, NEUTRAL_VENUE])

        render_info_note(
            "Team A and Team B are labels only. Host advantage applies only for a selected 2026 host country."
        )

    missing_fixture_teams = [team for team in [team_a, team_b] if model_team_name(team) not in teams]
    if missing_fixture_teams:
        st.warning(f"Fixture includes teams not found in the model snapshot: {', '.join(missing_fixture_teams)}.")
        st.stop()

    if team_a == team_b:
        st.warning("Choose two different teams.")
        st.stop()

    model_team_a = model_team_name(team_a)
    model_team_b = model_team_name(team_b)
    model_home_team, model_away_team, team_a_is_model_home, neutral, host_side = model_mapping_for_match(
        model_team_a, model_team_b, venue_country
    )
    host_status = host_status_text(host_side, team_a, team_b, venue_country)
    neutral_status = neutral_status_text(host_side)
    host_advantage_status = host_advantage_status_text(host_side, team_a, team_b)

    render_info_note(host_status)

    if selected_match_id is not None:
        render_team_news(selected_match_id, team_a, team_b)

    try:
        X = make_fixture_features(model_home_team, model_away_team, latest, neutral=neutral, tournament=tournament)
        outcome_artifact = load_outcome_artifact(outcome_model)
        outcome_probs = predict_outcome_from_artifact(X, outcome_model, outcome_artifact)
        home_xg, away_xg, scorelines, score_outcome = predict_score(X, max_goals=6)
    except Exception as exc:
        log_server_error("predictor prediction generation", exc)
        render_public_error()
        return

    labels = {
        "team_a_win": f"{team_a} win",
        "draw": "Draw",
        "team_b_win": f"{team_b} win",
    }
    outcome_display_probs = team_result_probabilities(outcome_probs, team_a_is_model_home)
    highest_probability_key = max(outcome_display_probs, key=outcome_display_probs.get)
    with st.container(border=True):
        render_section_header("Outcome Probabilities", f"Selected model: {outcome_model}", "Match forecast")
        prob_cols = st.columns(3)
        for i, key in enumerate(["team_a_win", "draw", "team_b_win"]):
            with prob_cols[i]:
                render_probability_card(labels[key], outcome_display_probs[key], is_top=(key == highest_probability_key))

    with st.container(border=True):
        render_section_header(
            "Scoreline Model",
            "Expected goals, score outcome probabilities and the most likely scorelines.",
            "Score prediction",
        )
        team_a_xg = home_xg if team_a_is_model_home else away_xg
        team_b_xg = away_xg if team_a_is_model_home else home_xg
        score_cols = st.columns(3)
        with score_cols[0]:
            render_score_card(f"Expected goals: {team_a}", f"{team_a_xg:.2f}", "Team A goal estimate")
        with score_cols[1]:
            render_score_card(f"Expected goals: {team_b}", f"{team_b_xg:.2f}", "Team B goal estimate")
        top_scoreline = scorelines.iloc[0]
        if team_a_is_model_home:
            most_likely_score = str(top_scoreline["scoreline"])
        else:
            most_likely_score = f"{top_scoreline['away_goals']}-{top_scoreline['home_goals']}"
        with score_cols[2]:
            render_score_card(
                f"Most likely score ({team_a}-{team_b})",
                most_likely_score,
                "Top scoreline from the score model",
            )

        render_section_header("Score Outcome Probabilities", "Derived from the goal model.", "Score model")
        score_prob_cols = st.columns(3)
        score_display_probs = team_result_probabilities(score_outcome, team_a_is_model_home)
        highest_score_probability_key = max(score_display_probs, key=score_display_probs.get)
        for i, key in enumerate(["team_a_win", "draw", "team_b_win"]):
            with score_prob_cols[i]:
                render_probability_card(
                    labels[key],
                    score_display_probs[key],
                    is_top=(key == highest_score_probability_key),
                )

        render_section_header("Top Scorelines", "Most likely scorelines from the goal model.", "Score table")
        top_scorelines = scorelines.head(10).copy()
        if not team_a_is_model_home:
            top_scorelines["scoreline"] = (
                top_scorelines["away_goals"].astype(str) + "-" + top_scorelines["home_goals"].astype(str)
            )
        top_scorelines["outcome"] = top_scorelines["outcome"].map(
            lambda outcome: model_outcome_label(outcome, team_a_is_model_home, team_a, team_b)
        )
        top_scorelines["probability"] = (top_scorelines["probability"] * 100).round(2)
        render_dark_table(top_scorelines[["scoreline", "outcome", "probability"]], hide_index=True)

        match_prediction_export = pd.DataFrame(
            [
                {
                    "Team A": team_a,
                    "Team B": team_b,
                    "Venue country": venue_country,
                    "Selected model": outcome_model,
                    "Team A win probability": outcome_display_probs["team_a_win"],
                    "Draw probability": outcome_display_probs["draw"],
                    "Team B win probability": outcome_display_probs["team_b_win"],
                    "Predicted scoreline": most_likely_score,
                    "Training cutoff date": TRAINING_CUTOFF_DATE,
                }
            ]
        )
        render_csv_download(
            match_prediction_export,
            "match_prediction.csv",
            "Download match prediction CSV",
            "download_match_prediction",
        )

    team_a_features = latest_by_team.loc[model_team_a]
    team_b_features = latest_by_team.loc[model_team_b]
    team_a_elo = float(team_a_features["current_elo"])
    team_b_elo = float(team_b_features["current_elo"])
    elo_diff = team_a_elo - team_b_elo
    team_a_ppg_last_5 = float(team_a_features["points_per_game_last_5"])
    team_b_ppg_last_5 = float(team_b_features["points_per_game_last_5"])
    team_a_goal_diff_last_5 = float(team_a_features["goal_diff_avg_last_5"])
    team_b_goal_diff_last_5 = float(team_b_features["goal_diff_avg_last_5"])
    team_a_win_rate_last_5 = float(team_a_features["win_rate_last_5"])
    team_b_win_rate_last_5 = float(team_b_features["win_rate_last_5"])

    with st.container(border=True):
        render_section_header(
            "Why This Prediction?",
            "Key model inputs and plain-English context for the selected matchup.",
            "Explanation",
        )
        render_section_header("Top Factors", "Per-match contributions are available for Logistic Regression models.", "Model view")
        contribution_table = local_logistic_contribution_table(
            model_label=outcome_model,
            artifact=outcome_artifact,
            fixture_features=X,
            outcome_probs=outcome_probs,
            team_a_is_model_home=team_a_is_model_home,
            top_n=10,
        )
        if contribution_table is None or contribution_table.empty:
            st.info(
                "Detailed per-match feature contributions are currently available for Logistic Regression models. "
                "Use Feature Importance for XGBoost-level global importance."
            )
        else:
            display_contribution_table = contribution_table.copy()
            display_contribution_table["Value"] = display_contribution_table["Value"].map(lambda value: f"{value:.3f}")
            display_contribution_table["Contribution"] = display_contribution_table["Contribution"].map(
                lambda value: f"{value:+.4f}"
            )
            render_dark_table(display_contribution_table.set_index("Feature"))

        explanation_table = pd.DataFrame(
            [
                {"Feature": f"Team A Elo ({team_a})", "Value": f"{team_a_elo:.0f}"},
                {"Feature": f"Team B Elo ({team_b})", "Value": f"{team_b_elo:.0f}"},
                {"Feature": "Elo difference (Team A - Team B)", "Value": f"{elo_diff:+.0f}"},
                {"Feature": f"Team A points per game last 5 ({team_a})", "Value": f"{team_a_ppg_last_5:.2f}"},
                {"Feature": f"Team B points per game last 5 ({team_b})", "Value": f"{team_b_ppg_last_5:.2f}"},
                {"Feature": f"Team A goal difference avg last 5 ({team_a})", "Value": f"{team_a_goal_diff_last_5:+.2f}"},
                {"Feature": f"Team B goal difference avg last 5 ({team_b})", "Value": f"{team_b_goal_diff_last_5:+.2f}"},
                {"Feature": f"Team A win rate last 5 ({team_a})", "Value": f"{team_a_win_rate_last_5 * 100:.0f}%"},
                {"Feature": f"Team B win rate last 5 ({team_b})", "Value": f"{team_b_win_rate_last_5 * 100:.0f}%"},
                {"Feature": "Neutral venue status", "Value": neutral_status},
                {"Feature": "Host advantage status", "Value": host_advantage_status},
            ]
        )
        render_section_header("Input Snapshot", "Core feature values passed into the prediction.", "Features")
        render_dark_table(explanation_table.set_index("Feature"))

        if all(column in latest.columns for column in LATEST_XG_COLUMNS):
            xg_explanation_table = pd.DataFrame(
                [
                    {"Feature": f"Team A xG for avg last 5 ({team_a})", "Value": f"{float(team_a_features['xg_for_avg_last_5']):.2f}"},
                    {"Feature": f"Team B xG for avg last 5 ({team_b})", "Value": f"{float(team_b_features['xg_for_avg_last_5']):.2f}"},
                    {
                        "Feature": f"Team A xG against avg last 5 ({team_a})",
                        "Value": f"{float(team_a_features['xg_against_avg_last_5']):.2f}",
                    },
                    {
                        "Feature": f"Team B xG against avg last 5 ({team_b})",
                        "Value": f"{float(team_b_features['xg_against_avg_last_5']):.2f}",
                    },
                ]
            )
            render_section_header("xG Snapshot", "Shown only when xG features are available.", "Optional layer")
            render_dark_table(xg_explanation_table.set_index("Feature"))
        else:
            st.info("xG features are not currently available for this model.")

        explanation_points = build_explanation_points(
            team_a=team_a,
            team_b=team_b,
            outcome_probs=outcome_display_probs,
            elo_diff=elo_diff,
            ppg_diff=team_a_ppg_last_5 - team_b_ppg_last_5,
            goal_diff_diff=team_a_goal_diff_last_5 - team_b_goal_diff_last_5,
            win_rate_diff=team_a_win_rate_last_5 - team_b_win_rate_last_5,
            neutral_status=neutral_status,
            host_advantage_status=host_advantage_status,
        )
        render_bullet_panel(explanation_points)


def predict_group_fixture(row: pd.Series, latest: pd.DataFrame, outcome_model: str) -> dict:
    team_a = str(row["team_a"])
    team_b = str(row["team_b"])
    venue_country = str(row["venue_country"])
    model_team_a = model_team_name(team_a)
    model_team_b = model_team_name(team_b)
    model_home_team, model_away_team, team_a_is_model_home, neutral, _ = model_mapping_for_match(
        model_team_a, model_team_b, venue_country
    )

    X = make_fixture_features(model_home_team, model_away_team, latest, neutral=neutral, tournament="FIFA World Cup")
    outcome_probs = team_result_probabilities(predict_outcome(X, model_name=outcome_model), team_a_is_model_home)
    _, _, scorelines, _ = predict_score(X, max_goals=6)
    top_scoreline = scorelines.iloc[0]

    if team_a_is_model_home:
        team_a_goals = int(top_scoreline["home_goals"])
        team_b_goals = int(top_scoreline["away_goals"])
    else:
        team_a_goals = int(top_scoreline["away_goals"])
        team_b_goals = int(top_scoreline["home_goals"])

    projected_key = max(outcome_probs, key=outcome_probs.get)
    projected_result = {
        "team_a_win": f"{team_a} win",
        "draw": "Draw",
        "team_b_win": f"{team_b} win",
    }[projected_key]

    return {
        "date": str(row["date"]),
        "team_a": team_a,
        "team_b": team_b,
        "venue": f"{row['venue_city']}, {venue_country}",
        "team_a_win_prob": outcome_probs["team_a_win"],
        "draw_prob": outcome_probs["draw"],
        "team_b_win_prob": outcome_probs["team_b_win"],
        "team_a_goals": team_a_goals,
        "team_b_goals": team_b_goals,
        "predicted_score": f"{team_a_goals}-{team_b_goals}",
        "projected_key": projected_key,
        "projected_result": projected_result,
    }


def group_teams(predictions: list[dict]) -> list[str]:
    return sorted(
        {prediction["team_a"] for prediction in predictions} | {prediction["team_b"] for prediction in predictions}
    )


def empty_group_standings(teams: list[str]) -> dict[str, dict[str, int | str]]:
    return {
        team: {
            "Team": team,
            "Played": 0,
            "Wins": 0,
            "Draws": 0,
            "Losses": 0,
            "Goals For": 0,
            "Goals Against": 0,
            "Goal Difference": 0,
            "Points": 0,
        }
        for team in teams
    }


def update_group_standings(
    standings: dict[str, dict[str, int | str]],
    team_a: str,
    team_b: str,
    team_a_goals: int,
    team_b_goals: int,
    result_key: str,
) -> None:
    standings[team_a]["Played"] += 1
    standings[team_b]["Played"] += 1
    standings[team_a]["Goals For"] += team_a_goals
    standings[team_a]["Goals Against"] += team_b_goals
    standings[team_b]["Goals For"] += team_b_goals
    standings[team_b]["Goals Against"] += team_a_goals

    if result_key == "team_a_win":
        standings[team_a]["Wins"] += 1
        standings[team_b]["Losses"] += 1
        standings[team_a]["Points"] += 3
    elif result_key == "team_b_win":
        standings[team_b]["Wins"] += 1
        standings[team_a]["Losses"] += 1
        standings[team_b]["Points"] += 3
    else:
        standings[team_a]["Draws"] += 1
        standings[team_b]["Draws"] += 1
        standings[team_a]["Points"] += 1
        standings[team_b]["Points"] += 1


def ranked_standings_rows(standings: dict[str, dict[str, int | str]]) -> list[dict[str, int | str]]:
    rows = []
    for standing in standings.values():
        row = dict(standing)
        row["Goal Difference"] = int(row["Goals For"]) - int(row["Goals Against"])
        rows.append(row)
    return sorted(rows, key=lambda row: (row["Points"], row["Goal Difference"], row["Goals For"]), reverse=True)


def standings_to_table(standings: dict[str, dict[str, int | str]]) -> pd.DataFrame:
    return pd.DataFrame(ranked_standings_rows(standings)).reset_index(drop=True)


def build_projected_group_table(predictions: list[dict]) -> pd.DataFrame:
    standings = empty_group_standings(group_teams(predictions))
    for prediction in predictions:
        update_group_standings(
            standings,
            prediction["team_a"],
            prediction["team_b"],
            int(prediction["team_a_goals"]),
            int(prediction["team_b_goals"]),
            prediction["projected_key"],
        )
    return standings_to_table(standings)


def adjust_scoreline_for_result(team_a_goals: int, team_b_goals: int, result_key: str) -> tuple[int, int]:
    if result_key == "team_a_win" and team_a_goals <= team_b_goals:
        team_a_goals = team_b_goals + 1
    elif result_key == "team_b_win" and team_b_goals <= team_a_goals:
        team_b_goals = team_a_goals + 1
    elif result_key == "draw":
        draw_goals = int((team_a_goals + team_b_goals) / 2)
        team_a_goals = draw_goals
        team_b_goals = draw_goals
    return team_a_goals, team_b_goals


def sample_result_key(prediction: dict, rng: random.Random) -> str:
    weights = [prediction["team_a_win_prob"], prediction["draw_prob"], prediction["team_b_win_prob"]]
    if sum(weights) <= 0:
        weights = [1, 1, 1]
    return rng.choices(GROUP_RESULT_KEYS, weights=weights, k=1)[0]


def simulate_group_once(predictions: list[dict], rng: random.Random) -> list[dict[str, int | str]]:
    standings = empty_group_standings(group_teams(predictions))
    for prediction in predictions:
        result_key = sample_result_key(prediction, rng)
        team_a_goals, team_b_goals = adjust_scoreline_for_result(
            int(prediction["team_a_goals"]),
            int(prediction["team_b_goals"]),
            result_key,
        )
        update_group_standings(
            standings,
            prediction["team_a"],
            prediction["team_b"],
            team_a_goals,
            team_b_goals,
            result_key,
        )
    return ranked_standings_rows(standings)


def run_monte_carlo_group_simulation(predictions: list[dict], simulation_count: int) -> pd.DataFrame:
    teams = group_teams(predictions)
    totals = {
        team: {
            "winner_count": 0,
            "top_2_count": 0,
            "third_count": 0,
            "points": 0,
            "goal_difference": 0,
        }
        for team in teams
    }
    rng = random.Random(2026)

    for _ in range(simulation_count):
        table = simulate_group_once(predictions, rng)
        for position, row in enumerate(table):
            team = row["Team"]
            totals[team]["winner_count"] += int(position == 0)
            totals[team]["top_2_count"] += int(position < 2)
            totals[team]["third_count"] += int(position == 2)
            totals[team]["points"] += int(row["Points"])
            totals[team]["goal_difference"] += int(row["Goal Difference"])

    results = pd.DataFrame(
        [
            {
                "Team": team,
                "Group winner %": values["winner_count"] / simulation_count * 100,
                "Top 2 %": values["top_2_count"] / simulation_count * 100,
                "3rd place %": values["third_count"] / simulation_count * 100,
                "Average points": values["points"] / simulation_count,
                "Average goal difference": values["goal_difference"] / simulation_count,
            }
            for team, values in totals.items()
        ]
    )
    return results.sort_values(
        ["Top 2 %", "Group winner %", "Average points", "Average goal difference"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def format_monte_carlo_results(results: pd.DataFrame) -> pd.DataFrame:
    display = results.copy()
    for column in ["Group winner %", "Top 2 %", "3rd place %"]:
        display[column] = display[column].map(lambda value: f"{value:.1f}%")
    for column in ["Average points", "Average goal difference"]:
        display[column] = display[column].map(lambda value: f"{value:.2f}")
    return display


def predict_fixtures_by_group(
    group_fixtures: pd.DataFrame,
    latest: pd.DataFrame,
    outcome_model: str,
) -> dict[str, list[dict]]:
    predictions_by_group = {}
    for group, fixtures in group_fixtures.groupby(group_fixtures["group"].astype(str), sort=True):
        predictions_by_group[group] = [
            predict_group_fixture(row, latest, outcome_model) for _, row in fixtures.iterrows()
        ]
    return predictions_by_group


def run_all_groups_simulation(
    predictions_by_group: dict[str, list[dict]],
    simulation_count: int,
) -> pd.DataFrame:
    team_groups = {
        team: group
        for group, predictions in predictions_by_group.items()
        for team in group_teams(predictions)
    }
    totals = {
        team: {
            "group": group,
            "winner_count": 0,
            "top_2_count": 0,
            "best_third_count": 0,
            "points": 0,
            "goal_difference": 0,
        }
        for team, group in team_groups.items()
    }
    rng = random.Random(2027)

    for _ in range(simulation_count):
        third_place_rows = []
        for group, predictions in predictions_by_group.items():
            table = simulate_group_once(predictions, rng)
            for position, row in enumerate(table):
                team = row["Team"]
                totals[team]["winner_count"] += int(position == 0)
                totals[team]["top_2_count"] += int(position < 2)
                totals[team]["points"] += int(row["Points"])
                totals[team]["goal_difference"] += int(row["Goal Difference"])
                if position == 2:
                    third_place_rows.append({**row, "Group": group})

        third_place_rows = sorted(
            third_place_rows,
            key=lambda row: (row["Points"], row["Goal Difference"], row["Goals For"]),
            reverse=True,
        )
        for row in third_place_rows[:8]:
            totals[row["Team"]]["best_third_count"] += 1

    results = pd.DataFrame(
        [
            {
                "Team": team,
                "Group": values["group"],
                "Group winner %": values["winner_count"] / simulation_count * 100,
                "Top 2 %": values["top_2_count"] / simulation_count * 100,
                "Best 3rd-place qualifier %": values["best_third_count"] / simulation_count * 100,
                "Round of 32 qualification %": (
                    values["top_2_count"] + values["best_third_count"]
                )
                / simulation_count
                * 100,
                "Average points": values["points"] / simulation_count,
                "Average goal difference": values["goal_difference"] / simulation_count,
            }
            for team, values in totals.items()
        ]
    )
    return results.sort_values(
        ["Round of 32 qualification %", "Top 2 %", "Group winner %"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def format_all_groups_results(results: pd.DataFrame) -> pd.DataFrame:
    display = results.copy()
    for column in [
        "Group winner %",
        "Top 2 %",
        "Best 3rd-place qualifier %",
        "Round of 32 qualification %",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.1f}%")
    for column in ["Average points", "Average goal difference"]:
        display[column] = display[column].map(lambda value: f"{value:.2f}")
    return display


def simulate_group_stage_qualifiers(
    predictions_by_group: dict[str, list[dict]],
    rng: random.Random,
) -> list[dict[str, int | str]]:
    qualified_rows = []
    third_place_rows = []

    for group, predictions in predictions_by_group.items():
        table = simulate_group_once(predictions, rng)
        for position, row in enumerate(table, start=1):
            qualified_row = {**row, "Group": group, "Group Position": position}
            if position <= 2:
                qualified_rows.append(qualified_row)
            elif position == 3:
                third_place_rows.append(qualified_row)

    third_place_rows = sorted(
        third_place_rows,
        key=lambda row: (row["Points"], row["Goal Difference"], row["Goals For"]),
        reverse=True,
    )
    return qualified_rows + third_place_rows[:8]


def projected_group_stage_qualifiers(predictions_by_group: dict[str, list[dict]]) -> list[dict[str, int | str]]:
    qualified_rows = []
    third_place_rows = []

    for group, predictions in predictions_by_group.items():
        table = build_projected_group_table(predictions)
        for position, row in table.iterrows():
            qualified_row = {**row.to_dict(), "Group": group, "Group Position": position + 1}
            if position < 2:
                qualified_rows.append(qualified_row)
            elif position == 2:
                third_place_rows.append(qualified_row)

    third_place_rows = sorted(
        third_place_rows,
        key=lambda row: (row["Points"], row["Goal Difference"], row["Goals For"]),
        reverse=True,
    )
    return qualified_rows + third_place_rows[:8]


def knockout_win_probability(
    team_a: str,
    team_b: str,
    latest: pd.DataFrame,
    outcome_model: str,
    outcome_artifact,
    cache: dict[tuple[str, str], float],
) -> float:
    key = (team_a, team_b)
    if key in cache:
        return cache[key]

    X = make_fixture_features(
        model_team_name(team_a),
        model_team_name(team_b),
        latest,
        neutral=1,
        tournament="FIFA World Cup",
    )
    outcome_probs = predict_outcome_from_artifact(X, outcome_model, outcome_artifact)
    team_probs = team_result_probabilities(outcome_probs, team_a_is_model_home=True)
    win_total = team_probs["team_a_win"] + team_probs["team_b_win"]
    team_a_prob = 0.5 if win_total <= 0 else team_probs["team_a_win"] / win_total
    cache[key] = team_a_prob
    return team_a_prob


def knockout_match_projection(
    match_id: int,
    team_a: str,
    team_b: str,
    latest: pd.DataFrame,
    outcome_model: str,
    outcome_artifact,
    cache: dict[tuple[str, str], float],
) -> dict[str, int | str | float]:
    team_a_prob = knockout_win_probability(team_a, team_b, latest, outcome_model, outcome_artifact, cache)
    team_b_prob = 1 - team_a_prob
    return {
        "Match number": match_id,
        "Team A": team_a,
        "Team B": team_b,
        "Team A win %": team_a_prob * 100,
        "Team B win %": team_b_prob * 100,
        "Projected winner": team_a if team_a_prob >= team_b_prob else team_b,
    }


def simulate_knockout_pair(
    team_a: str,
    team_b: str,
    latest: pd.DataFrame,
    outcome_model: str,
    outcome_artifact,
    cache: dict[tuple[str, str], float],
    rng: random.Random,
) -> str:
    team_a_prob = knockout_win_probability(team_a, team_b, latest, outcome_model, outcome_artifact, cache)
    return team_a if rng.random() < team_a_prob else team_b


def simulate_knockout_path(
    path: list[tuple[int, int, int]],
    match_winners: dict[int, str],
    latest: pd.DataFrame,
    outcome_model: str,
    outcome_artifact,
    cache: dict[tuple[str, str], float],
    rng: random.Random,
) -> list[str]:
    winners = []
    for match_id, left_match, right_match in path:
        winner = simulate_knockout_pair(
            match_winners[left_match],
            match_winners[right_match],
            latest,
            outcome_model,
            outcome_artifact,
            cache,
            rng,
        )
        match_winners[match_id] = winner
        winners.append(winner)
    return winners


def project_knockout_path(
    path: list[tuple[int, int, int]],
    match_winners: dict[int, str],
    latest: pd.DataFrame,
    outcome_model: str,
    outcome_artifact,
    cache: dict[tuple[str, str], float],
) -> list[dict[str, int | str | float]]:
    rows = []
    for match_id, left_match, right_match in path:
        row = knockout_match_projection(
            match_id,
            match_winners[left_match],
            match_winners[right_match],
            latest,
            outcome_model,
            outcome_artifact,
            cache,
        )
        match_winners[match_id] = str(row["Projected winner"])
        rows.append(row)
    return rows


def build_sample_bracket_path(
    predictions_by_group: dict[str, list[dict]],
    latest: pd.DataFrame,
    outcome_model: str,
) -> tuple[dict[str, pd.DataFrame], bool]:
    outcome_artifact = load_outcome_artifact(outcome_model)
    cache: dict[tuple[str, str], float] = {}
    qualified_rows = projected_group_stage_qualifiers(predictions_by_group)
    round_32_matches, used_third_place_fallback = build_round_of_32_matches(qualified_rows)
    match_winners: dict[int, str] = {}
    round_32_rows = []

    for match in round_32_matches:
        row = knockout_match_projection(
            int(match["match"]),
            str(match["team_a"]),
            str(match["team_b"]),
            latest,
            outcome_model,
            outcome_artifact,
            cache,
        )
        match_winners[int(match["match"])] = str(row["Projected winner"])
        round_32_rows.append(row)

    if len(round_32_matches) != 16:
        empty_round = pd.DataFrame(
            columns=["Match number", "Team A", "Team B", "Team A win %", "Team B win %", "Projected winner"]
        )
        return {
            "Round of 32": pd.DataFrame(round_32_rows),
            "Round of 16": empty_round.copy(),
            "Quarter-finals": empty_round.copy(),
            "Semi-finals": empty_round.copy(),
            "Final": empty_round.copy(),
        }, True

    round_16_rows = project_knockout_path(ROUND_OF_16_PATH, match_winners, latest, outcome_model, outcome_artifact, cache)
    quarter_final_rows = project_knockout_path(
        QUARTER_FINAL_PATH, match_winners, latest, outcome_model, outcome_artifact, cache
    )
    semi_final_rows = project_knockout_path(SEMI_FINAL_PATH, match_winners, latest, outcome_model, outcome_artifact, cache)

    final_match_id, left_match, right_match = FINAL_PATH
    final_rows = [
        knockout_match_projection(
            final_match_id,
            match_winners[left_match],
            match_winners[right_match],
            latest,
            outcome_model,
            outcome_artifact,
            cache,
        )
    ]

    return {
        "Round of 32": pd.DataFrame(round_32_rows),
        "Round of 16": pd.DataFrame(round_16_rows),
        "Quarter-finals": pd.DataFrame(quarter_final_rows),
        "Semi-finals": pd.DataFrame(semi_final_rows),
        "Final": pd.DataFrame(final_rows),
    }, used_third_place_fallback


def run_knockout_simulation(
    predictions_by_group: dict[str, list[dict]],
    latest: pd.DataFrame,
    outcome_model: str,
    simulation_count: int,
) -> tuple[pd.DataFrame, int]:
    team_groups = {
        team: group
        for group, predictions in predictions_by_group.items()
        for team in group_teams(predictions)
    }
    totals = {
        team: {
            "group": group,
            "round_32": 0,
            "round_16": 0,
            "quarter_final": 0,
            "semi_final": 0,
            "final": 0,
            "champion": 0,
        }
        for team, group in team_groups.items()
    }
    rng = random.Random(2028)
    outcome_artifact = load_outcome_artifact(outcome_model)
    knockout_cache: dict[tuple[str, str], float] = {}
    third_place_fallback_count = 0

    for _ in range(simulation_count):
        qualified_rows = simulate_group_stage_qualifiers(predictions_by_group, rng)
        for row in qualified_rows:
            totals[row["Team"]]["round_32"] += 1

        if len(qualified_rows) != 32:
            continue

        round_32_matches, used_third_place_fallback = build_round_of_32_matches(qualified_rows)
        third_place_fallback_count += int(used_third_place_fallback)
        if len(round_32_matches) != 16:
            continue

        match_winners = {}
        round_16_teams = []
        for match in round_32_matches:
            winner = simulate_knockout_pair(
                str(match["team_a"]),
                str(match["team_b"]),
                latest,
                outcome_model,
                outcome_artifact,
                knockout_cache,
                rng,
            )
            match_winners[int(match["match"])] = winner
            round_16_teams.append(winner)
        for team in round_16_teams:
            totals[team]["round_16"] += 1

        quarter_final_teams = simulate_knockout_path(
            ROUND_OF_16_PATH, match_winners, latest, outcome_model, outcome_artifact, knockout_cache, rng
        )
        for team in quarter_final_teams:
            totals[team]["quarter_final"] += 1

        semi_final_teams = simulate_knockout_path(
            QUARTER_FINAL_PATH, match_winners, latest, outcome_model, outcome_artifact, knockout_cache, rng
        )
        for team in semi_final_teams:
            totals[team]["semi_final"] += 1

        final_teams = simulate_knockout_path(
            SEMI_FINAL_PATH, match_winners, latest, outcome_model, outcome_artifact, knockout_cache, rng
        )
        for team in final_teams:
            totals[team]["final"] += 1

        final_match_id, left_match, right_match = FINAL_PATH
        champion = simulate_knockout_pair(
            match_winners[left_match],
            match_winners[right_match],
            latest,
            outcome_model,
            outcome_artifact,
            knockout_cache,
            rng,
        )
        match_winners[final_match_id] = champion
        totals[champion]["champion"] += 1

    results = pd.DataFrame(
        [
            {
                "Team": team,
                "Group": values["group"],
                "Round of 32 %": values["round_32"] / simulation_count * 100,
                "Round of 16 %": values["round_16"] / simulation_count * 100,
                "Quarter-final %": values["quarter_final"] / simulation_count * 100,
                "Semi-final %": values["semi_final"] / simulation_count * 100,
                "Final %": values["final"] / simulation_count * 100,
                "Champion %": values["champion"] / simulation_count * 100,
            }
            for team, values in totals.items()
        ]
    )
    results = results.sort_values(
        ["Champion %", "Final %", "Semi-final %", "Quarter-final %"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return results, third_place_fallback_count


def format_knockout_results(results: pd.DataFrame) -> pd.DataFrame:
    display = results.copy()
    for column in [
        "Round of 32 %",
        "Round of 16 %",
        "Quarter-final %",
        "Semi-final %",
        "Final %",
        "Champion %",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.1f}%")
    return display


def format_bracket_round(round_table: pd.DataFrame) -> pd.DataFrame:
    display = round_table.copy()
    for column in ["Team A win %", "Team B win %"]:
        display[column] = display[column].map(lambda value: f"{value:.1f}%")
    return display


def render_group_simulation(latest: pd.DataFrame, teams: list[str]) -> None:
    render_section_header(
        "Group Simulation",
        "Project one group, run Monte Carlo simulations, and estimate all-groups qualification chances.",
        "Tournament view",
    )

    fixtures, fixture_warning = safe_load_world_cup_fixtures("group simulation fixture loading")
    if fixture_warning:
        st.warning(fixture_warning)
        return
    if fixtures is None or fixtures.empty:
        st.warning("No fixtures were found in the fixture CSV.")
        return

    group_fixtures = group_stage_fixtures(fixtures)
    if group_fixtures.empty:
        st.warning("No Group Stage fixtures were found in the fixture CSV.")
        return

    fixture_summary = validate_group_stage_fixtures(fixtures)
    with st.container(border=True):
        render_section_header("Fixture Check", "Each 4-team group should have 4 teams and 6 fixtures.", "Data quality")
        render_dark_table(fixture_summary, hide_index=True)

        groups = sorted(group_fixtures["group"].dropna().astype(str).unique().tolist())
        if not groups:
            st.warning("Group Stage fixtures are missing group labels.")
            return

        col1, col2 = st.columns(2)
        with col1:
            selected_group = st.selectbox("Group", groups)
        with col2:
            outcome_model = st.selectbox("Group simulation outcome model", available_outcome_models())

        selected_fixtures = group_fixtures[group_fixtures["group"].astype(str) == selected_group].copy()
        if selected_fixtures.empty:
            st.warning(f"No fixtures found for Group {selected_group}.")
            return

        selected_summary = fixture_summary[fixture_summary["Group"].astype(str) == selected_group]
        if not selected_summary.empty and selected_summary.iloc[0]["Status"] == "Incomplete":
            row = selected_summary.iloc[0]
            st.warning(
                f"Group {selected_group} fixture data is incomplete "
                f"({row['Number of teams']} teams, {row['Number of fixtures']} fixtures). "
                "Simulation results may be unreliable."
            )

        with st.expander(f"Fixtures for Group {selected_group}"):
            render_dark_table(
                selected_fixtures[
                    ["date", "kickoff_time", "team_a", "team_b", "venue_city", "venue_country", "stage"]
                ].rename(
                    columns={
                        "date": "Date",
                        "kickoff_time": "Kickoff time",
                        "team_a": "Team A",
                        "team_b": "Team B",
                        "venue_city": "Venue city",
                        "venue_country": "Venue country",
                        "stage": "Stage",
                    }
                ),
                hide_index=True,
            )

    missing_teams = sorted(
        {
            team
            for _, fixture in selected_fixtures.iterrows()
            for team in [str(fixture["team_a"]), str(fixture["team_b"])]
            if model_team_name(team) not in teams
        }
    )
    if missing_teams:
        st.warning(f"Some fixture teams are not available in the model snapshot: {', '.join(missing_teams)}.")
        return

    try:
        predictions = [predict_group_fixture(row, latest, outcome_model) for _, row in selected_fixtures.iterrows()]
        group_table = build_projected_group_table(predictions)
    except Exception as exc:
        log_server_error("selected group prediction generation", exc)
        render_public_error()
        return

    with st.container(border=True):
        render_section_header(
            f"Projected Group {selected_group}",
            "Deterministic table based on each fixture's highest-probability result.",
            "Single projection",
        )
        if len(group_table) >= 2:
            st.success(f"Likely direct qualifiers: {group_table.iloc[0]['Team']} and {group_table.iloc[1]['Team']}.")
        if len(group_table) >= 3:
            st.info(f"Possible qualifier, depending on third-place rules: {group_table.iloc[2]['Team']}.")

        selected_group_projection = group_table[
            [
                "Team",
                "Played",
                "Wins",
                "Draws",
                "Losses",
                "Goals For",
                "Goals Against",
                "Goal Difference",
                "Points",
            ]
        ].copy()
        render_dark_table(selected_group_projection, hide_index=True)
        render_csv_download(
            selected_group_projection,
            "selected_group_projection.csv",
            "Download selected group projection CSV",
            "download_selected_group_projection",
        )

    with st.container(border=True):
        render_section_header("Match-by-match Predictions", "Fixture-level probabilities and projected scorelines.", "Fixtures")
        match_rows = [
            {
                "Date": prediction["date"],
                "Team A": prediction["team_a"],
                "Team B": prediction["team_b"],
                "Venue": prediction["venue"],
                "Team A win %": f"{prediction['team_a_win_prob'] * 100:.1f}%",
                "Draw %": f"{prediction['draw_prob'] * 100:.1f}%",
                "Team B win %": f"{prediction['team_b_win_prob'] * 100:.1f}%",
                "Predicted score": prediction["predicted_score"],
                "Projected result": prediction["projected_result"],
            }
            for prediction in predictions
        ]
        group_match_predictions = pd.DataFrame(match_rows)
        render_dark_table(group_match_predictions, hide_index=True)
        render_csv_download(
            group_match_predictions,
            "group_match_predictions.csv",
            "Download group match predictions CSV",
            "download_group_match_predictions",
        )

    with st.container(border=True):
        render_section_header("Monte Carlo Simulation", "Repeatedly sample outcomes from model probabilities.", "Uncertainty")
        render_info_note(
            "Monte Carlo simulation captures uncertainty better than a single deterministic projection."
        )
        simulation_count = st.selectbox(
            "Number of simulations",
            SIMULATION_OPTIONS,
            index=1,
            format_func=lambda value: f"{value:,}",
        )
        try:
            monte_carlo_results = run_monte_carlo_group_simulation(predictions, simulation_count)
        except Exception as exc:
            log_server_error("selected group Monte Carlo simulation", exc)
            render_public_error()
            return
        render_dark_table(format_monte_carlo_results(monte_carlo_results), hide_index=True)
        render_csv_download(
            monte_carlo_results,
            "selected_group_monte_carlo.csv",
            "Download selected group Monte Carlo CSV",
            "download_selected_group_monte_carlo",
        )

    with st.container(border=True):
        render_section_header(
            "All Groups Simulation",
            "Run the entire group stage thousands of times and estimate Round of 32 qualification.",
            "Full field",
        )
        render_info_note(
            "All Groups Simulation captures uncertainty better than a single projected table."
        )
        if fixture_summary["Status"].eq("Incomplete").any():
            st.warning("Some group-stage fixture data is incomplete. All-groups simulation may be unreliable.")

        all_group_missing_teams = sorted(
            {
                team
                for _, fixture in group_fixtures.iterrows()
                for team in [str(fixture["team_a"]), str(fixture["team_b"])]
                if model_team_name(team) not in teams
            }
        )
        if all_group_missing_teams:
            st.warning(
                "Some group-stage fixture teams are not available in the model snapshot: "
                f"{', '.join(all_group_missing_teams)}."
            )
            return

        all_col1, all_col2 = st.columns(2)
        with all_col1:
            all_group_simulation_count = st.selectbox(
                "Number of all-groups simulations",
                SIMULATION_OPTIONS,
                index=1,
                format_func=lambda value: f"{value:,}",
            )
        with all_col2:
            all_group_outcome_model = st.selectbox("All groups outcome model", available_outcome_models())

        all_group_filter_options = ["All teams"] + [f"Group {group}" for group in groups]
        all_group_filter = st.selectbox("Group filter", all_group_filter_options)

        try:
            predictions_by_group = predict_fixtures_by_group(group_fixtures, latest, all_group_outcome_model)
            all_groups_results = run_all_groups_simulation(predictions_by_group, all_group_simulation_count)
        except Exception as exc:
            log_server_error("all groups prediction generation", exc)
            render_public_error()
            return
        if all_group_filter != "All teams":
            selected_filter_group = all_group_filter.replace("Group ", "", 1)
            all_groups_results = all_groups_results[all_groups_results["Group"].astype(str) == selected_filter_group]

        render_dark_table(format_all_groups_results(all_groups_results), hide_index=True)
        render_csv_download(
            all_groups_results,
            "all_groups_simulation.csv",
            "Download all groups simulation CSV",
            "download_all_groups_simulation",
        )


def render_knockout_simulation(latest: pd.DataFrame, teams: list[str]) -> None:
    render_section_header(
        "Knockout Simulation",
        "FIFA-style Round of 32 slots with simplified third-place assignment.",
        "Bracket path",
    )

    fixtures, fixture_warning = safe_load_world_cup_fixtures("knockout fixture loading")
    if fixture_warning:
        st.warning(fixture_warning)
        return
    if fixtures is None or fixtures.empty:
        st.warning("No fixtures were found in the fixture CSV.")
        return

    group_fixtures = group_stage_fixtures(fixtures)
    if group_fixtures.empty:
        st.warning("No Group Stage fixtures were found in the fixture CSV.")
        return

    fixture_summary = validate_group_stage_fixtures(fixtures)
    if fixture_summary["Status"].eq("Incomplete").any():
        st.warning("Some group-stage fixture data is incomplete. Knockout simulation may be unreliable.")

    missing_teams = sorted(
        {
            team
            for _, fixture in group_fixtures.iterrows()
            for team in [str(fixture["team_a"]), str(fixture["team_b"])]
            if model_team_name(team) not in teams
        }
    )
    if missing_teams:
        st.warning(f"Some group-stage fixture teams are not available in the model snapshot: {', '.join(missing_teams)}.")
        return

    with st.container(border=True):
        render_section_header("Knockout Controls", "Choose the model and simulation count.", "Setup")
        col1, col2 = st.columns(2)
        with col1:
            outcome_model = st.selectbox("Knockout outcome model", available_outcome_models())
        with col2:
            simulation_count = st.selectbox(
                "Number of knockout simulations",
                SIMULATION_OPTIONS,
                index=1,
                format_func=lambda value: f"{value:,}",
            )

    try:
        predictions_by_group = predict_fixtures_by_group(group_fixtures, latest, outcome_model)
        knockout_results, third_place_fallback_count = run_knockout_simulation(
            predictions_by_group, latest, outcome_model, simulation_count
        )
    except Exception as exc:
        log_server_error("knockout prediction generation", exc)
        render_public_error()
        return

    with st.container(border=True):
        render_section_header("Tournament Probabilities", "How often each team reaches each stage.", "Monte Carlo")
        if knockout_results["Round of 32 %"].max() == 0:
            st.warning("The current fixture data did not produce a 32-team knockout field.")
        if third_place_fallback_count:
            st.warning(
                "Some simulations used fallback third-place assignment because no eligible team was available "
                "for one or more Round of 32 slots."
            )

        render_dark_table(format_knockout_results(knockout_results), hide_index=True)
        render_csv_download(
            knockout_results,
            "knockout_probabilities.csv",
            "Download knockout probabilities CSV",
            "download_knockout_probabilities",
        )

    with st.container(border=True):
        render_section_header("Sample Bracket Path", "One deterministic projected bracket using the current model.", "Bracket")
        render_info_note(
            "The probability table is more reliable than this single path because football outcomes are uncertain."
        )
        try:
            bracket_rounds, sample_used_fallback = build_sample_bracket_path(predictions_by_group, latest, outcome_model)
        except Exception as exc:
            log_server_error("sample bracket prediction generation", exc)
            render_public_error()
            return
        if sample_used_fallback:
            st.warning("This sample bracket used fallback third-place assignment for at least one Round of 32 slot.")

        for round_name, round_table in bracket_rounds.items():
            render_section_header(round_name, "Projected winners use adjusted knockout win probabilities.", "Round")
            render_dark_table(format_bracket_round(round_table), hide_index=True)

        sample_bracket_tables = [
            round_table.assign(Round=round_name)[
                ["Round", "Match number", "Team A", "Team B", "Team A win %", "Team B win %", "Projected winner"]
            ]
            for round_name, round_table in bracket_rounds.items()
            if not round_table.empty
        ]
        if sample_bracket_tables:
            sample_bracket_export = pd.concat(sample_bracket_tables, ignore_index=True)
            render_csv_download(
                sample_bracket_export,
                "sample_bracket_path.csv",
                "Download sample bracket path CSV",
                "download_sample_bracket_path",
            )


def render_model_evaluation() -> None:
    render_section_header(
        "Model Evaluation",
        "Saved metrics, baseline-vs-xG comparisons, and model feature importance.",
        "Diagnostics",
    )

    outcome_metrics_path = ARTIFACTS_DIR / "outcome_metrics.json"
    score_metrics_path = ARTIFACTS_DIR / "score_metrics.json"
    missing_metrics = [path.name for path in [outcome_metrics_path, score_metrics_path] if not path.exists()]
    if missing_metrics:
        log_server_error("model evaluation metrics loading", FileNotFoundError(", ".join(missing_metrics)))
        render_public_error()
        return

    score_metrics = read_metrics(score_metrics_path)
    outcome_model_metrics = {}
    for label, filename in OUTCOME_METRICS_FILES.items():
        path = ARTIFACTS_DIR / filename
        if path.exists():
            outcome_model_metrics[label] = read_metrics(path)

    if not outcome_model_metrics:
        st.warning("No outcome model metrics were found.")
        return

    def metric_row(label: str, key: str, formatter=str) -> dict[str, str]:
        row = {"Metric": label}
        for model_label, metrics in outcome_model_metrics.items():
            row[model_label] = formatter(metrics.get(key))
        return row

    def test_range(metrics: dict) -> str:
        return f"{metrics.get('test_start_date', 'Not available')} to {metrics.get('test_end_date', 'Not available')}"

    def format_count(value: object) -> str:
        if value is None:
            return "Not available"
        return f"{int(value):,}"

    accuracy_candidates = [
        (label, metrics.get("accuracy")) for label, metrics in outcome_model_metrics.items()
        if metrics.get("accuracy") is not None
    ]
    log_loss_candidates = [
        (label, metrics.get("log_loss")) for label, metrics in outcome_model_metrics.items()
        if metrics.get("log_loss") is not None
    ]
    best_accuracy = max(accuracy_candidates, key=lambda item: item[1]) if accuracy_candidates else None
    best_log_loss = min(log_loss_candidates, key=lambda item: item[1]) if log_loss_candidates else None

    with st.container(border=True):
        render_section_header("Performance Snapshot", "Current saved evaluation metrics.", "Metrics")
        metric_cards = st.columns(4)
        with metric_cards[0]:
            render_kpi_card("Training cutoff", TRAINING_CUTOFF_DATE, "No leakage window")
        with metric_cards[1]:
            render_kpi_card("Outcome models", str(len(outcome_model_metrics)), "Metrics loaded")
        with metric_cards[2]:
            if best_accuracy:
                render_kpi_card("Best accuracy", format_percent(best_accuracy[1]), best_accuracy[0])
        with metric_cards[3]:
            if best_log_loss:
                render_kpi_card("Best log loss", format_number(best_log_loss[1], digits=3), best_log_loss[0])

        outcome_rows = [
            metric_row("Training cutoff date", "training_cutoff_date", lambda value: str(value or TRAINING_CUTOFF_DATE)),
            metric_row("Outcome model name", "model", lambda value: str(value or "Not available")),
            {"Metric": "Test date range", **{label: test_range(metrics) for label, metrics in outcome_model_metrics.items()}},
            metric_row("Number of train rows", "train_rows", format_count),
            metric_row("Number of validation rows", "validation_rows", format_count),
            metric_row("Number of test rows", "test_rows", format_count),
            metric_row("Accuracy", "accuracy", format_percent),
            metric_row("Log loss", "log_loss", lambda value: format_number(value, digits=3)),
            metric_row("Brier score", "brier_score", lambda value: format_number(value, digits=3)),
        ]
        render_dark_table(pd.DataFrame(outcome_rows).set_index("Metric"))

        score_rows = [
            {
                "Metric": "Score model RMSE for Team A/Home goals",
                "Value": format_number(score_metrics.get("home_goals_rmse"), digits=3),
            },
            {
                "Metric": "Score model RMSE for Team B/Away goals",
                "Value": format_number(score_metrics.get("away_goals_rmse"), digits=3),
            },
        ]
        render_dark_table(pd.DataFrame(score_rows).set_index("Metric"))

        render_bullet_panel(
            [
                "Accuracy = how often the model picked the correct result class.",
                "Log loss = how good the probability estimates are; lower is better.",
                "Brier score = how close the predicted probabilities are to the actual result; lower is better.",
                "RMSE = average goal prediction error; lower is better.",
            ]
        )

    with st.container(border=True):
        render_section_header("Baseline vs xG Model Comparison", "Compare baseline models against xG-enhanced versions.", "Model stack")
        comparison_rows = []
        for label, filename in BASELINE_XG_COMPARISON_FILES.items():
            path = ARTIFACTS_DIR / filename
            if not path.exists():
                continue
            metrics = read_metrics(path)
            comparison_rows.append(
                {
                    "Model": label,
                    "xG used?": "Yes" if metrics.get("xg_features_used") else "No",
                    "Accuracy": format_percent(metrics.get("accuracy")),
                    "Log loss": format_number(metrics.get("log_loss"), digits=3),
                    "Train rows": f"{int(metrics.get('train_rows', 0)):,}",
                    "Test rows": f"{int(metrics.get('test_rows', 0)):,}",
                }
            )

        if comparison_rows:
            render_dark_table(pd.DataFrame(comparison_rows).set_index("Model"))
            render_info_note(
                "Higher accuracy is better. Lower log loss is better. If xG improves log loss, probability quality improved."
            )
        else:
            st.info("Baseline vs xG comparison metrics are not available yet. Run the full pipeline to create them.")

    with st.container(border=True):
        render_section_header(
            "Feature Importance",
            "Which inputs the model relies on most. This does not prove causation.",
            "Interpretability",
        )

        importance_tables: dict[str, pd.DataFrame] = {}
        skipped_importance_models: list[str] = []
        for label in available_outcome_models():
            try:
                artifact = load_outcome_artifact(label)
                table = feature_importance_table(label, artifact)
            except Exception:
                table = None

            if table is not None and not table.empty:
                importance_tables[label] = table
            else:
                skipped_importance_models.append(label)

        if not importance_tables:
            st.warning("Feature importance is unavailable for the saved outcome model artifacts.")
        else:
            selected_importance_model = st.selectbox("Feature importance model", list(importance_tables))
            importance_table = importance_tables[selected_importance_model].copy()
            display_importance_table = importance_table.copy()
            display_importance_table["Importance"] = display_importance_table["Importance"].map(lambda value: f"{value:.4f}")
            render_dark_table(display_importance_table.set_index("Feature"))
            render_csv_download(
                importance_table,
                "feature_importance.csv",
                "Download feature importance CSV",
                "download_feature_importance",
            )

            if importance_table["Feature"].str.contains("xg", case=False, regex=False).any():
                st.info("xG features appear in the top features, so xG is contributing to this model's predictions.")

        if skipped_importance_models:
            st.caption(
                "Feature importance could not be safely extracted for: "
                + ", ".join(skipped_importance_models)
            )


def render_methodology() -> None:
    render_section_header(
        "Methodology",
        "A concise view of the data, assumptions, model stack, and fixture manager.",
        "Transparency",
    )

    card_rows = [
        [
            (
                "Data",
                [
                    "Historical international results",
                    "Local World Cup fixture CSV",
                    "Optional xG and team-news CSVs",
                ],
            ),
            (
                "Training",
                [
                    f"Cutoff: {TRAINING_CUTOFF_DATE}",
                    "Only pre-cutoff matches are used",
                    "Rolling features avoid same-match leakage",
                ],
            ),
            (
                "Models",
                [
                    "Logistic Regression baseline",
                    "XGBoost comparison model",
                    "Calibrated probability variants",
                ],
            ),
        ],
        [
            (
                "Simulation",
                [
                    "Score model predicts likely goals",
                    "Monte Carlo samples match outcomes",
                    "Knockout uses simplified FIFA-style slots",
                ],
            ),
            (
                "Limitations",
                [
                    "No live API updates",
                    "Team news is context only",
                    "Predictions are probabilities, not certainties",
                ],
            ),
            (
                "Next",
                [
                    "Add official stadium and kickoff data",
                    "Improve xG and player availability coverage",
                    "Add exact third-place bracket mapping",
                ],
            ),
        ],
    ]

    with st.container(border=True):
        render_section_header("Project Assumptions", "What the app uses and what it deliberately leaves out.", "Overview")
        for row in card_rows:
            columns = st.columns(3)
            for column, (title, items) in zip(columns, row):
                with column:
                    render_method_card(title, items)

    with st.container(border=True):
        render_section_header(
            "Fixture Data Manager",
            "Inspect, validate, and replace the local World Cup fixture CSV.",
            "Data tools",
        )
        fixtures, fixture_warning = safe_load_world_cup_fixtures("methodology fixture loading")
        if fixture_warning:
            st.warning(fixture_warning)
            fixtures = None

        if fixtures is not None and not fixtures.empty:
            quality_summary, fixtures_per_group, quality_warnings = fixture_quality_checks(fixtures)
            render_info_note("Current fixture file is loaded.")
            render_dark_table(quality_summary.set_index("Check"))

            render_section_header("Fixtures Per Group", "Fixture counts from the current CSV.", "Validation")
            render_dark_table(fixtures_per_group, hide_index=True)

            for warning in quality_warnings:
                st.warning(warning)

            with st.expander("Current fixture table"):
                render_dark_table(fixtures, hide_index=True)

        uploaded_file = st.file_uploader("Upload replacement fixture CSV", type=["csv"])
        if uploaded_file is None:
            return

        uploaded_fixtures, upload_error = validate_uploaded_fixtures(uploaded_file)
        if upload_error:
            st.error(upload_error)
            return

        st.success("Uploaded CSV passed required column and basic date validation.")
        uploaded_summary, uploaded_per_group, uploaded_warnings = fixture_quality_checks(uploaded_fixtures)
        render_section_header("Uploaded Fixture Checks", "Validation results before replacement.", "Upload")
        render_dark_table(uploaded_summary.set_index("Check"))
        render_dark_table(uploaded_per_group, hide_index=True)
        for warning in uploaded_warnings:
            st.warning(warning)

        if st.button("Replace fixture CSV", key="replace_fixture_csv"):
            backup_path = PROCESSED_DIR / "world_cup_2026_fixtures_backup_before_upload.csv"
            if FIXTURES_PATH.exists():
                shutil.copy2(FIXTURES_PATH, backup_path)
            uploaded_fixtures.to_csv(FIXTURES_PATH, index=False)
            st.success(
                "Saved replacement fixture CSV and created a backup."
            )
            render_section_header("Updated Fixture Table", "Replacement data saved locally.", "Saved")
            render_dark_table(uploaded_fixtures, hide_index=True)


st.set_page_config(page_title="World Cup Predictor", page_icon=":soccer:", layout="wide")

inject_dashboard_css()
render_hero()

latest = load_startup_snapshot()
try:
    teams = sorted(latest["team"].dropna().unique().tolist())
except Exception as exc:
    log_server_error("team snapshot validation", exc)
    render_public_error()
    st.stop()

predictor_tab, group_simulation_tab, evaluation_tab, methodology_tab = st.tabs(
    ["Predictor", "Group Simulation", "Model Evaluation", "Methodology"]
)
with predictor_tab:
    safe_render_section("Predictor tab", render_predictor, latest, teams)
with group_simulation_tab:
    safe_render_section("Group Simulation tab", render_group_simulation, latest, teams)
    with st.expander("Knockout Simulation", expanded=False):
        safe_render_section("Knockout Simulation section", render_knockout_simulation, latest, teams)
with evaluation_tab:
    safe_render_section("Model Evaluation tab", render_model_evaluation)
with methodology_tab:
    safe_render_section("Methodology tab", render_methodology)
