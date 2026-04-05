# euroml
Rule-aware EuroMillions research project.

## Data update workflow

`download.py` is the bulk archive loader.

Examples:

```bash
uv run python download.py --year 2026
uv run python download.py --start-year 2024 --end-year 2026
uv run python download.py --all-history
```

`daily_load.py` is the cron-safe updater.
It refreshes the latest local archive year through the current year, so it keeps the yearly `*_eml.json` files current and handles year rollover cleanly.

Examples:

```bash
uv run python daily_load.py
uv run python daily_load.py --skip-test-csv
```

## Prediction engine

The forecasting engine is designed as a chronological research system, not a free-form neural predictor.
It enforces EuroMillions rules by draw date, builds candidate-level features, trains small-data classical models, and generates legal ranked tickets.

Backtest the engine on held-out years:

```bash
uv run python prediction_engine.py backtest \
  --strategy baseline \
  --train-end 2024-12-31 \
  --test-start 2025-01-01 \
  --test-end 2025-12-31 \
  --mode frozen \
  --output reports/backtest_2025.json
```

Predict the next unseen draw:

```bash
uv run python prediction_engine.py predict-next \
  --strategy baseline \
  --output reports/next_draw_prediction.json
```

Compare candidate strategies on the same held-out window:

```bash
uv run python prediction_engine.py compare-strategies \
  --train-end 2024-12-31 \
  --test-start 2025-01-01 \
  --test-end 2025-12-31 \
  --mode frozen \
  --strategies baseline multi_history
```

Shortcut scripts:

```bash
./scripts/backtest_2025.sh
./scripts/backtest_2025_rolling.sh
./scripts/backtest_2026.sh
./scripts/compare_strategies.sh
./scripts/predict_next.sh
```

The engine also maintains a local structured state file `.engine.mem.json` so it can reuse run context safely without hiding logic or contaminating the dataset.

## Research automation

Use the research runner to refresh the compact comparison reports and the latest next-draw prediction:

```bash
bash scripts/research_cycle.sh
```

This writes committed outputs into `research_outputs/latest/`:

- `compare_2025.json`
- `compare_2026.json`
- `next_draw_prediction.json`
- `engine_state.json`
- `metadata.json`
- `summary.md`

The workflow [`.github/workflows/research.yml`](.github/workflows/research.yml) triggers on each code push and ignores `research_outputs/**`, so the automated result commits do not loop forever.

After the workflow is pushed to GitHub you can track it with `gh`:

```bash
gh run list --workflow research.yml
gh run watch
```
