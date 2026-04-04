# euroml
Simple raw data for testing RNN

## Data update workflow

`download.py` is the bulk archive loader.

Examples:

```bash
uv run python euroml/download.py --year 2026
uv run python euroml/download.py --start-year 2024 --end-year 2026
uv run python euroml/download.py --all-history
```

`daily_load.py` is the cron-safe updater.
It refreshes the latest local archive year through the current year, so it keeps the yearly `*_eml.json` files current and handles year rollover cleanly.

Examples:

```bash
uv run python euroml/daily_load.py
uv run python euroml/daily_load.py --skip-test-csv
```
