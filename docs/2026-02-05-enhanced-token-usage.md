# Enhanced Token Usage Reporting Implementation Plan

> **For Gemini:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task.
> **Mandate:** ALL file operations must use absolute paths.
> **Mandate:** ALWAYS verify workspace type before VCS operations.

**Goal:** Enhance `token_usage.py` to support granular model breakdown in totals and summary statistics, and add comprehensive date filtering options using only standard Python libraries.

**Investigation & Analysis:**
- `scripts/token_usage.py` current structure:
    - `aggregate_usage()`: Recursively searches for session JSONs. Hardcodes path.
    - `calculate_cost()`: Handles tiered pricing and model-specific rates.
    - `print_report()`: Prints the table. Aggregates totals but doesn't break them down by model even if `--model` is provided.
    - `print_summary_statistics()`: Aggregates everything into one daily total.
- `scripts/token-usage_test.py` current structure:
    - Mocks `aggregate_usage` by redefining its logic inside the test. This is fragile.

**Strategic Approach:**
- **Refactor for Testability**: Modify `aggregate_usage` to accept a path.
- **Date Range Helpers**: Implement a robust date range calculator for "yesterday", "this-week", etc.
- **Enhanced Aggregation**: Ensure totals and summaries respect the `--model` flag.
- **Strict Standard Library Usage**: Ensure no `pip install` is required.

**Anticipated Challenges & Considerations:**
- Tiered pricing ($2 vs $4 for Pro) is currently calculated *per message*. This is correct for large context messages.
- Date range boundaries (e.g., "last-week") need careful calculation relative to "today".
- Formatting long model names in the summary table.

**Tech Stack:** Python 3 (Standard Library: `argparse`, `datetime`, `json`, `pathlib`, `collections`, `unittest`).

---

### Task 1: Refactor for Testability & Enhance Aggregation

**Files:**
- Modify: `/usr/local/google/home/rkj/misc/geminiusage/scripts/token_usage.py`
- Modify: `/usr/local/google/home/rkj/misc/geminiusage/scripts/token-usage_test.py`

**Step 1: Audit & Precedent Analysis (Look Before You Leap)**
- Analyze `scripts/token-usage_test.py`. It currently re-implements `aggregate_usage` logic.
- Plan: Update `tu.aggregate_usage(base_dir=None)` to allow passing a directory.

**Step 2: RED (Test First)**
- Update `test_aggregation` to pass the temp directory directly to `aggregate_usage`.
- Add a test case for `calculate_cost` with various models to ensure coverage.

**Step 3: GREEN (Implementation)**
- Update `aggregate_usage` in `token_usage.py`.
- Improve error handling in JSON loading.

**Step 4: Verification (Standard Workflow)**
- Test: `python3 scripts/token-usage_test.py`

---

### Task 2: Implement Date Filtering Logic

**Files:**
- Modify: `/usr/local/google/home/rkj/misc/geminiusage/scripts/token_usage.py`
- Modify: `/usr/local/google/home/rkj/misc/geminiusage/scripts/token-usage_test.py`

**Step 1: Audit & Precedent Analysis**
- Current date logic uses `datetime.now().strftime("%Y-%m-%d")`.
- Need a function `get_date_range(filter_name)` that returns `(start_date, end_date)`.

**Step 2: RED (Test First)**
- Write tests for `get_date_range` ensuring "yesterday", "this-week", "last-month" return correct strings.

**Step 3: GREEN (Implementation)**
- Add flags to `argparse`: `--yesterday`, `--this-week`, `--last-week`, `--this-month`, `--last-month`, `--date-range`.
- Logic for "this-week": Monday to today.
- Logic for "last-week": Previous Monday to previous Sunday.
- Implement `filter_stats(stats, start_date, end_date)`.

**Step 4: Verification**
- Test: `python3 scripts/token-usage_test.py`

---

### Task 3: Model Breakdown in Totals & Summaries

**Files:**
- Modify: `/usr/local/google/home/rkj/misc/geminiusage/scripts/token_usage.py`
- Modify: `/usr/local/google/home/rkj/misc/geminiusage/scripts/token-usage_test.py`

**Step 1: Audit & Precedent Analysis**
- `print_report`'s `grand_total_tokens` is a single integer.
- `print_summary_statistics` uses `daily_totals` (aggregated across models).

**Step 2: RED (Test First)**
- Add a test case that checks if `print_report(show_models=True)` output contains model-specific totals.
- Add a test case for `print_summary_statistics(show_models=True)`.

**Step 3: GREEN (Implementation)**
- Update `print_report`: if `show_models`, group grand totals by model.
- Update `print_summary_statistics`: if `show_models`, loop through models and print stats for each.

**Step 4: Verification**
- Test: `python3 scripts/token-usage_test.py`

---

### Task 4: Final Polishing & Best Practices

**Files:**
- Modify: `/usr/local/google/home/rkj/misc/geminiusage/scripts/token_usage.py`

**Step 1: Audit**
- Check for any leftover hardcoded paths or `print` statements.
- Ensure 80-char line limits (standard practice).

**Step 2: Verification**
- Run a full manual test if possible (or mock all components).
- Final check on `argparse` descriptions.

---

### Task 5: Refine Reporting Format & Add Costs to Summaries

**Files:**
- Modify: `/usr/local/google/home/rkj/misc/geminiusage/scripts/token_usage.py`
- Modify: `/usr/local/google/home/rkj/misc/geminiusage/scripts/token-usage_test.py`

**Step 1: Audit & Analysis**
- Current truncation: `label = f"TOTALS ({model[:20]})"`. Remove truncation.
- `print_summary_statistics` currently only prints token counts. Needs to track and print `cost` as well.

**Step 2: RED (Test First)**
- Update `test_print_report_with_models` to check for full model name in totals.
- Update `test_print_summary_statistics_with_models` to check for cost strings.

**Step 3: GREEN (Implementation)**
- Remove `[:20]` truncation in `print_report`.
- Update `print_summary_statistics` to aggregate and display costs for all time, last 7 days, last 30 days, and per model.

**Step 4: Verification (Standard Workflow)**
- Test: `python3 scripts/token-usage_test.py`
