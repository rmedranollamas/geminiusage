# Gemini Usage Tracker (GEMINI.md)

This project is a lightweight, high-performance Python utility designed to track, aggregate, and visualize token usage and costs for Google Gemini AI models. It operates by parsing session logs (JSON format), applying tiered pricing logic, and providing both a CLI reporter and an interactive Terminal User Interface (TUI).

## Project Overview

- **Core Logic (`scripts/token_usage.py`)**: Handles recursive file scanning, JSON parsing, tiered cost calculation, and persistent caching (`usage_cache.json`).
- **Interactive Dashboard (`scripts/tui.py`)**: A curses-based TUI with auto-refresh (30s interval), live countdowns, and dynamic filtering.
- **Data Persistence**: Uses a local cache to avoid re-parsing unchanged session files, making it suitable for frequent execution (e.g., every 30 seconds).
- **Configuration**: Supports custom pricing overrides via `~/.gemini/pricing.json`.

## Building and Running

The project relies exclusively on the Python 3 standard library (no external `pip` dependencies required).

### Commands

- **Run CLI Report**:
  ```bash
  python3 scripts/token_usage.py [dir] [--today|--this-week|--model]
  ```
- **Run TUI Dashboard**:
  ```bash
  python3 scripts/tui.py [dir]
  ```
- **Run Tests**:
  ```bash
  python3 scripts/token-usage_test.py
  python3 scripts/tui_test.py
  ```

## Development Conventions

- **Standard Library First**: Prioritize Python's standard library modules (`curses`, `argparse`, `dataclasses`, `pathlib`, `json`, `datetime`) to keep the project zero-dependency and easy to distribute.
- **Performance**: Any changes to data ingestion must respect the `usage_cache.json` logic (checking `mtime` before parsing).
- **Testing Strategy**:
  - Use `unittest` and `unittest.mock` for core logic and UI state testing.
  - Logic tests should use `tempfile.TemporaryDirectory` to simulate filesystem environments.
- **Pricing Logic**: Tiered pricing is sensitive to context thresholds (default 200k tokens). Ensure `calculate_cost` remains the single source of truth for pricing math.
- **Configuration**: Custom pricing should be loaded via `load_config` and merged with hardcoded defaults.

## Key Files

- `scripts/token_usage.py`: The "engine" - parsing, aggregation, and CLI.
- `scripts/tui.py`: The "dashboard" - interactive presentation and auto-refresh logic.
- `scripts/token-usage_test.py`: Unit tests for aggregation and cost calculations.
- `scripts/tui_test.py`: Mocked curses tests for TUI state and rendering.
- `README.md`: General user documentation.
- `CONTRIBUTING.md`: Guidelines for development and technical standards.
