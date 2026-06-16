# Codex task prompts for this repo

Use these prompts inside Codex after opening this folder.

## Prompt 1: Inspect and run

```text
Inspect this repository. Run the full pipeline, fix any runtime errors, and keep the architecture simple. Do not add unnecessary abstractions. Make sure artifacts are created and the Streamlit app can load them.
```

## Prompt 2: Add xG later

```text
Add an optional StatsBomb Open Data ingestion module for 2018 and 2022 World Cup events. Compute rolling xG-for and xG-against features where available. Do not use same-match xG as an input feature for that match. Keep the xG features optional so the base model still runs without them.
```

## Prompt 3: Improve evaluation

```text
Improve model evaluation using time-based splits. Evaluate on 2018 World Cup and 2022 World Cup separately. Report accuracy, log loss, Brier score, and calibration curves. Add a short markdown report to artifacts/.
```

## Prompt 4: Improve app UX

```text
Improve the Streamlit app UI. Add probability bars, top 10 scorelines, expected goals, and a small explanation of the strongest features driving the prediction.
```

## Prompt 5: Prepare for 2026 World Cup

```text
Add a fixtures ingestion layer that can load 2026 World Cup fixtures from a CSV file. Let the app predict every fixture in the file and export predictions as CSV.
```
