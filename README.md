# World Cup 2026 Predictor

A Streamlit app for exploring World Cup 2026 match predictions, scorelines, group-stage simulations, knockout simulations, and model evaluation.

The project is designed to run locally without paid APIs. It uses historical international match results, Elo-style ratings, recent form features, optional local xG data, and optional local team-news files.

## Main Features

- Match predictor with Team A / Team B wording and neutral or host-country venue logic.
- Outcome probabilities from Logistic Regression and XGBoost models.
- Baseline-vs-xG model comparison.
- Score prediction with likely scorelines.
- Predictor explanation panel with key features and Logistic Regression local contributions.
- Group Simulation tab for deterministic and Monte Carlo group projections.
- All-groups simulation for Round of 32 qualification estimates.
- Knockout simulation tools inside the Group Simulation tab.
- Model Evaluation tab with metrics and feature importance.
- Optional xG feature layer from local CSV data.
- Local player availability and lineup/team-news display.
- Fixture data manager for inspecting or replacing the local fixture CSV.
- Downloadable CSV exports for key outputs.

## Quick Start on Windows PowerShell

From the project folder, run:

```powershell
.\run_app.ps1
```

The script will:

- Create `.venv` if needed.
- Install `requirements.txt`.
- Run `scripts/smoke_test.py` if available.
- Start Streamlit with `app/streamlit_app.py`.

## Retrain Models

To rebuild features and retrain the saved models:

```powershell
.\train_models.ps1
```

Model artifacts and metrics are saved in:

```text
artifacts/
```

## Manual Commands

If you prefer to run commands directly:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
.\.venv\Scripts\python.exe scripts\run_full_pipeline.py
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

## Data Files

The app uses these local files:

```text
data/processed/world_cup_2026_fixtures.csv
data/processed/xg_team_match_features.csv
data/processed/player_availability.csv
data/processed/lineups.csv
```

Expected fixture columns:

```text
match_id,date,kickoff_time,group,team_a,team_b,venue_country,venue_city,stage
```

Expected xG columns:

```text
date,team,opponent,xg_for,xg_against,non_penalty_xg_for,non_penalty_xg_against
```

Expected player availability columns:

```text
match_id,team,player,status,reason,importance
```

Expected lineup columns:

```text
match_id,team,player,position,is_starter
```

Optional StatsBomb-style event JSON files can be placed under:

```text
data/raw/statsbomb/events/
```

Then run:

```powershell
.\.venv\Scripts\python.exe scripts\build_xg_features.py
```

## Methodology Summary

- Historical international match results are cleaned and filtered before the training cutoff date.
- The no-leakage cutoff is defined in `src/config.py` as `TRAINING_CUTOFF_DATE`.
- Elo and recent-form features are calculated from matches before each prediction target.
- Most World Cup fixtures are treated as neutral.
- Host advantage is applied only when Team A or Team B is playing in its own 2026 host country: United States, Mexico, or Canada.
- Outcome models predict home/team-A-side win, draw, or away/team-B-side win probabilities.
- Score models estimate likely goals separately from the outcome model.
- Optional xG features are used only when local xG data exists.
- Lineup and player availability data is displayed for context only and is not currently included in the model.
- Group and knockout simulations sample outcomes from model probabilities.
- Knockout simulations currently use FIFA-style Round of 32 slots with simplified third-place assignment.

## Limitations

- Predictions are probabilistic, not certain.
- No live fixture, lineup, injury, or odds API is used.
- Player availability and lineup data is not yet part of the model features.
- xG quality depends on local coverage and usable match dates.
- Some fixture kickoff times or venue cities may be placeholders unless manually updated.
- Knockout third-place assignment is simplified until exact official mapping is added.
- No xG, lineup, or injury live updates are fetched automatically.

## Future Improvements

- Add exact official kickoff times, stadiums, and fixture updates.
- Add a live fixture or lineup API.
- Add richer xG coverage and validation.
- Add player availability, injuries, and lineup strength into model features.
- Add exact FIFA Annex C third-place bracket mapping.
- Add SHAP or another local explanation method for XGBoost.
- Add a knockout bracket export or printable report view.

## Project Structure

```text
app/
  streamlit_app.py
scripts/
  build_xg_features.py
  run_full_pipeline.py
  smoke_test.py
src/
  config.py
  data.py
  elo.py
  features.py
  fixtures.py
  predict.py
  train_outcome_model.py
  train_score_model.py
  train_xgboost_model.py
artifacts/
data/
  processed/
  raw/
```
