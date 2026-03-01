# Contributing to Gemini Usage Tracker

Thank you for your interest in contributing! This project is a lightweight utility designed for speed and reliability.

## Workflow

1. **Fork the repo** and create your branch from `main`.
1. **Implement changes** following the existing style (4-space indentation, descriptive names).
1. **Add tests** for any new logic in `scripts/token-usage_test.py` or UI changes in `scripts/tui_test.py`.
1. **Verify types and linting** if you have `ruff` or `mypy` installed.
1. **Submit a Pull Request**.

## Technical Standards

- **Standard Library First**: Avoid adding external dependencies unless absolutely necessary.
- **Performance**: Use the caching mechanism in `token_usage.py` for any new parsing logic.
- **Safety**: Ensure JSON parsing is robust and handles missing/null fields gracefully.

## Bug Reports

Please include a sample (redacted) session JSON file if the bug is related to parsing errors.
