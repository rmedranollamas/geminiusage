#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

# Default patterns
FLASH_PATTERNS = ["flash"]
PRO_PATTERNS = ["pro"]


def load_config():
    """Loads custom model mappings from config files."""
    config = {
        "flash_patterns": list(FLASH_PATTERNS),
        "pro_patterns": list(PRO_PATTERNS),
    }
    # Check ~/.gemini/pricing.json
    p = Path.home() / ".gemini" / "pricing.json"
    if p.exists():
        try:
            with p.open("r") as f:
                data = json.load(f)
                config["flash_patterns"].extend(data.get("flash_patterns", []))
                config["pro_patterns"].extend(data.get("pro_patterns", []))
        except Exception:
            pass
    return config


CONFIG = load_config()


def calculate_cost(model, input_tokens, cached_tokens, output_tokens):
    """Calculates cost based on model type and tiered pricing."""
    model_lower = model.lower()
    context_size = input_tokens + cached_tokens

    # Default to Gemini 3 Pro pricing
    if context_size <= 200_000:
        i_rate, c_rate, o_rate = 2.00, 0.20, 12.00
    else:
        i_rate, c_rate, o_rate = 4.00, 0.40, 18.00

    # Model specific overrides
    if "gemini-3-flash" in model_lower:
        i_rate, c_rate, o_rate = 0.50, 0.05, 3.00
    elif "gemini-2.5-pro" in model_lower:
        if context_size <= 200_000:
            i_rate, c_rate, o_rate = 1.25, 0.125, 10.00
        else:
            i_rate, c_rate, o_rate = 2.50, 0.25, 15.00
    elif "gemini-2.5-flash-lite" in model_lower:
        i_rate, c_rate, o_rate = 0.10, 0.01, 0.40
    elif "gemini-2.5-flash" in model_lower:
        i_rate, c_rate, o_rate = 0.30, 0.03, 2.50
    elif "gemini-2.0-flash-lite" in model_lower:
        i_rate, c_rate, o_rate = 0.075, 0.0, 0.30  # Cache not available
    elif "gemini-2.0-flash" in model_lower:
        i_rate, c_rate, o_rate = 0.10, 0.025, 0.40
    elif "flash" in model_lower:
        # General Flash fallback (using 3-flash rates)
        i_rate, c_rate, o_rate = 0.50, 0.05, 3.00

    return (
        input_tokens * i_rate + cached_tokens * c_rate + output_tokens * o_rate
    ) / 1_000_000


def aggregate_usage():
    # Use real user home or resolve to the symlink
    home = Path.home()
    gemini_dir = home / ".gemini"
    # If symlink doesn't exist or is invalid, fallback to the direct path
    if not gemini_dir.exists():
        gemini_dir = Path("/usr/local/google/home/niobium/Code/dotgemini")

    tmp_dir = gemini_dir / "tmp"

    # stats[date][model] = {sessions: set(), input: 0, cached: 0, output: 0, cost: 0.0}
    stats = defaultdict(
        lambda: defaultdict(
            lambda: {
                "sessions": set(),
                "input": 0,
                "cached": 0,
                "output": 0,
                "cost": 0.0,
            }
        )
    )

    for session_file in tmp_dir.glob("*/chats/session-*.json"):
        try:
            with session_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            session_id = data.get("sessionId") or session_file.stem
            start_time = data.get("startTime", "")

            date = start_time.split("T")[0] if "T" in start_time else "unknown"

            messages = data.get("messages", [])
            for msg in messages:
                if msg.get("type") == "gemini":
                    model = msg.get("model", "unknown")
                    tokens = msg.get("tokens", {})

                    inp = tokens.get("input", 0)
                    cache = tokens.get("cached", 0)
                    out = tokens.get("output", 0) + tokens.get("thoughts", 0)

                    # Calculate cost per message to handle tiered thresholds correctly
                    cost = calculate_cost(model, inp, cache, out)

                    m_stats = stats[date][model]
                    m_stats["sessions"].add(session_id)
                    m_stats["input"] += inp
                    m_stats["cached"] += cache
                    m_stats["output"] += out
                    m_stats["cost"] += cost

        except Exception:
            continue

    return stats


def print_report(stats, show_models=False, today_only=False):
    if not stats:
        print("No usage data found.")
        return

    today_str = datetime.now().strftime("%Y-%m-%d")

    if today_only:
        if today_str not in stats:
            print(f"No usage data found for today ({today_str}).")
            return
        # Create a filtered version of stats containing only today
        stats = {today_str: stats[today_str]}

    # Header
    model_header = f"{'MODEL':<25} " if show_models else ""
    header = f"{'DATE':<12} {model_header}{'SESS':<5} {'INPUT':>12} {'CACHED':>12} {'OUTPUT':>12} {'TOTAL':>12} {'COST':>10}"
    line_len = len(header) + 2
    print(header)
    print("-" * line_len)

    grand_total_tokens = 0
    grand_total_cost = 0.0

    # Sort by date
    for date in sorted(stats.keys()):
        display_date = f"{date}*" if date == today_str else f"{date:<11} "
        if not show_models:
            # Aggregate all models for this date
            day_input = 0
            day_cached = 0
            day_output = 0
            day_sessions = set()
            day_cost = 0.0

            for model, s in stats[date].items():
                day_input += s["input"]
                day_cached += s["cached"]
                day_output += s["output"]
                day_sessions.update(s["sessions"])
                day_cost += s["cost"]

            total = day_input + day_cached + day_output
            print(
                f"{display_date:<12} {len(day_sessions):<5} "
                f"{day_input:>12,} {day_cached:>12,} {day_output:>12,} "
                f"{total:>12,} ${day_cost:>8.2f}"
            )

            grand_total_tokens += total
            grand_total_cost += day_cost
        else:
            for model in sorted(stats[date].keys()):
                s = stats[date][model]
                total = s["input"] + s["cached"] + s["output"]

                print(
                    f"{display_date:<12} {model[:25]:<25} {len(s['sessions']):<5} "
                    f"{s['input']:>12,} {s['cached']:>12,} {s['output']:>12,} "
                    f"{total:>12,} ${s['cost']:>8.2f}"
                )

                grand_total_tokens += total
                grand_total_cost += s["cost"]

    print("-" * line_len)
    if today_str in stats:
        print("* Note: Today's count is not complete (there can be more done! :)")
    total_label = "TOTALS"
    offset = 44 if show_models else 18
    print(
        f"{total_label:<{offset}} {grand_total_tokens:>50,} ${grand_total_cost:>8.2f}"
    )


def print_summary_statistics(stats):
    if not stats:
        return

    today = datetime.now().date()
    all_dates = sorted(
        [
            datetime.strptime(d, "%Y-%m-%d").date()
            for d in stats.keys()
            if d != "unknown"
        ]
    )
    if not all_dates:
        return

    # Total tokens per day (aggregated across models)
    daily_totals = defaultdict(int)
    for date, models in stats.items():
        if date == "unknown":
            continue
        d_obj = datetime.strptime(date, "%Y-%m-%d").date()
        for model, s in models.items():
            daily_totals[d_obj] += s["input"] + s["cached"] + s["output"]

    total_days = len(daily_totals)
    grand_total = sum(daily_totals.values())
    avg_per_day = grand_total / total_days if total_days > 0 else 0

    # Last N days calculations
    def get_period_stats(days):
        cutoff = today - timedelta(days=days)
        period_data = [v for k, v in daily_totals.items() if k >= cutoff]
        return sum(period_data), len(period_data)

    last_7_total, last_7_days_count = get_period_stats(7)
    last_30_total, last_30_days_count = get_period_stats(30)

    print("\nSUMMARY STATISTICS")
    print("-" * 30)
    print(f"{'Average tokens / day:':<25} {int(avg_per_day):>15,}")
    print(f"{'Last 7 days total:':<25} {last_7_total:>15,}")
    print(f"{'Last 7 days average:':<25} {int(last_7_total / 7):>15,}")
    print(f"{'Last 30 days total:':<25} {last_30_total:>15,}")
    print(f"{'Last 30 days average:':<25} {int(last_30_total / 30):>15,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Gemini token usage and costs."
    )
    parser.add_argument(
        "--model", action="store_true", help="Show breakdown per model."
    )
    parser.add_argument(
        "--today", action="store_true", help="Only show usage for today."
    )
    args = parser.parse_args()

    stats = aggregate_usage()
    print_report(stats, show_models=args.model, today_only=args.today)
    if not args.today:
        print_summary_statistics(stats)
