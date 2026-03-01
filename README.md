# Gemini Usage Tracker

A lightweight Python utility to track, aggregate, and visualize token usage and costs for Google Gemini AI models. It parses session logs, applies tiered pricing logic, and provides both a CLI reporter and an interactive Terminal User Interface (TUI).

## Features

- **High-Performance Aggregation**: Recursively scans session logs with persistent caching (`usage_cache.json`) for near-instant results on subsequent runs.
- **Interactive TUI**: A `curses`-based dashboard with auto-refresh (every 30s), live countdowns, and scrollable tables.
- **Flexible Pricing**: Supports tiered pricing (e.g., higher rates for >200k context) and user-defined overrides via `pricing.json`.
- **Comprehensive Reporting**: Filter by date ranges (`today`, `this-week`, `last-month`, etc.) or custom YYYY-MM-DD ranges.
- **Model Breakdown**: Toggle between summary and per-model views.

## Installation

The tool relies almost exclusively on the Python standard library.

1. Clone the repository:

   ```bash
   git clone https://github.com/rmedranollamas/geminiusage.git
   cd geminiusage
   ```

1. Ensure you have Python 3.7+ installed.

## Usage

### CLI Reporter

Run the core script to get a text-based report:

```bash
# Show usage for today
python3 scripts/token_usage.py --today

# Show per-model breakdown for this week
python3 scripts/token_usage.py --this-week --model

# Point to a custom session directory
python3 scripts/token_usage.py /path/to/my/sessions --today
```

### Interactive TUI

Launch the interactive dashboard:

```bash
python3 scripts/tui.py

# Or with a custom directory
python3 scripts/tui.py /path/to/my/sessions
```

**TUI Shortcuts:**

- `Q`: Quit
- `R`: Manual Refresh
- `M`: Toggle Model Breakdown
- `F`: Change Date Filter
- `P`: Edit Pricing Configuration
- `UP/DOWN`: Scroll Table

## Configuration

Custom model pricing can be defined in `~/.gemini/pricing.json`.

```json
{
  "models": {
    "my-custom-model": [1.0, 0.1, 5.0],
    "tiered-pro-model": {
      "small_context": [2.0, 0.2, 12.0],
      "large_context": [4.0, 0.4, 18.0],
      "context_threshold": 200000
    }
  }
}
```

*Rates are USD per 1 million tokens in the format `[input, cached, output]`. Throught tokens are included in the output count.*

## Testing

Run the test suite to verify logic and UI behavior:

```bash
python3 scripts/token-usage_test.py
python3 scripts/tui_test.py
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
