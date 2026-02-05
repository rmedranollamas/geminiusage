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


def aggregate_usage(base_dir=None):
    if base_dir:
        tmp_dir = Path(base_dir)
    else:
        # Use real user home
        gemini_dir = Path.home() / ".gemini"
        tmp_dir = gemini_dir / "tmp"

    # stats[date][model] = {
    #   sessions: set(), input: 0, cached: 0, output: 0, cost: 0.0
    # }
    def session_factory():
        return {
            "sessions": set(),
            "input": 0,
            "cached": 0,
            "output": 0,
            "cost": 0.0,
        }

    stats = defaultdict(lambda: defaultdict(session_factory))

    if not tmp_dir.exists():
        return stats

    for session_file in tmp_dir.glob("**/session-*.json"):
        try:
            with session_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                continue

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

                    # Calculate cost per message to handle tiered thresholds
                    # correctly.
                    cost = calculate_cost(model, inp, cache, out)

                    m_stats = stats[date][model]
                    m_stats["sessions"].add(session_id)
                    m_stats["input"] += inp
                    m_stats["cached"] += cache
                    m_stats["output"] += out
                    m_stats["cost"] += cost

        except (json.JSONDecodeError, IOError, KeyError):
            continue
        except Exception:
            # Catch-all for other unexpected issues but keep going
            continue

    return stats


def get_date_range(filter_name, today=None):
    """Returns (start_date, end_date) strings for a given filter."""
    if today is None:
        today = datetime.now().date()

    if filter_name == "yesterday":
        yesterday = today - timedelta(days=1)
        return yesterday.strftime("%Y-%m-%d"), yesterday.strftime("%Y-%m-%d")

    elif filter_name == "this-week":
        # Monday is 0, Sunday is 6
        start = today - timedelta(days=today.weekday())
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    elif filter_name == "last-week":
        # Last Monday to last Sunday
        end = today - timedelta(days=today.weekday() + 1)
        start = end - timedelta(days=6)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    elif filter_name == "this-month":
        start = today.replace(day=1)
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    elif filter_name == "last-month":
        # First day of this month
        first_of_this_month = today.replace(day=1)
        # Last day of last month
        end = first_of_this_month - timedelta(days=1)
        # First day of last month
        start = end.replace(day=1)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    elif ":" in filter_name:
        parts = filter_name.split(":")
        if len(parts) == 2:
            return parts[0], parts[1]

    return None, None


def filter_stats(stats, start_date, end_date):
    """Filters stats by date range (inclusive)."""
    filtered = {}
    for date, models in stats.items():
        if date == "unknown":
            continue
        if start_date <= date <= end_date:
            filtered[date] = models
    return filtered


def print_report(stats, show_models=False, today_only=False, raw=False):
    today_str = datetime.now().strftime("%Y-%m-%d")

    if today_only:
        if today_str not in stats:
            # Initialize empty stats for today
            stats = {
                today_str: {
                    "unknown": {
                        "sessions": set(),
                        "input": 0,
                        "cached": 0,
                        "output": 0,
                        "cost": 0.0,
                    }
                }
            }
        else:
            stats = {today_str: stats[today_str]}

    if not stats:
        if raw:
            print("0")
        else:
            print("No usage data found.")
        return

    grand_total_tokens = 0
    grand_total_cost = 0.0
    model_grand_totals = defaultdict(lambda: {"tokens": 0, "cost": 0.0})

    if raw:
        for date in stats:
            for model in stats[date]:
                s = stats[date][model]
                grand_total_tokens += s["input"] + s["cached"] + s["output"]
        print(grand_total_tokens)
        return

    # Header
    model_header = f"{'MODEL':<25} " if show_models else ""
    header = (
        f"{'DATE':<12} {model_header}{'SESS':<5} {'INPUT':>12} "
        f"{'CACHED':>12} {'OUTPUT':>12} {'TOTAL':>12} {'COST':>10}"
    )
    line_len = len(header) + 2
    print(header)
    print("-" * line_len)

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
                    f"{display_date:<12} {model:<25} "
                    f"{len(s['sessions']):<5} {s['input']:>12,} "
                    f"{s['cached']:>12,} {s['output']:>12,} "
                    f"{total:>12,} ${s['cost']:>8.2f}"
                )

                grand_total_tokens += total
                grand_total_cost += s["cost"]
                model_grand_totals[model]["tokens"] += total
                model_grand_totals[model]["cost"] += s["cost"]

    print("-" * line_len)
    if today_str in stats:
        msg = "* Note: Today's count is not complete (there can be more!)"
        print(msg)
    
    if show_models and len(model_grand_totals) > 1:
        for model in sorted(model_grand_totals.keys()):
            m_stats = model_grand_totals[model]
            label = f"TOTALS ({model})"
            print(
                f"{label:<44} {m_stats['tokens']:>50,} "
                f"${m_stats['cost']:>8.2f}"
            )
        print("-" * line_len)

    total_label = "TOTALS (ALL)" if show_models else "TOTALS"
    offset = 44 if show_models else 18
    print(
        f"{total_label:<{offset}} {grand_total_tokens:>50,} "
        f"${grand_total_cost:>8.2f}"
    )


def print_summary_statistics(stats, show_models=False):
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

    # Aggregate by day
    daily_token_totals = defaultdict(int)
    daily_cost_totals = defaultdict(float)
    # Aggregate by model
    model_totals = defaultdict(
        lambda: {"tokens": 0, "cost": 0.0, "days": set()}
    )

    for date, models in stats.items():
        if date == "unknown":
            continue
        d_obj = datetime.strptime(date, "%Y-%m-%d").date()
        for model, s in models.items():
            tokens = s["input"] + s["cached"] + s["output"]
            cost = s["cost"]
            
            daily_token_totals[d_obj] += tokens
            daily_cost_totals[d_obj] += cost
            
            model_totals[model]["tokens"] += tokens
            model_totals[model]["cost"] += cost
            model_totals[model]["days"].add(d_obj)

    total_days = len(daily_token_totals)
    grand_total_tokens = sum(daily_token_totals.values())
    grand_total_cost = sum(daily_cost_totals.values())
    
    avg_tokens_per_day = (
        grand_total_tokens / total_days if total_days > 0 else 0
    )
    avg_cost_per_day = grand_total_cost / total_days if total_days > 0 else 0

    # Last N days calculations
    def get_period_stats(days):
        cutoff = today - timedelta(days=days)
        period_tokens = [
            v for k, v in daily_token_totals.items() if k >= cutoff
        ]
        period_costs = [v for k, v in daily_cost_totals.items() if k >= cutoff]
        return sum(period_tokens), sum(period_costs)

    last_7_tokens, last_7_cost = get_period_stats(7)
    last_30_tokens, last_30_cost = get_period_stats(30)

    print("\nSUMMARY STATISTICS")
    print("-" * 30)
    print(f"{'Total tokens:':<25} {grand_total_tokens:>15,}")
    print(f"{'Total cost:':<25} ${grand_total_cost:>14.2f}")
    print(f"{'Average tokens / day:':<25} {int(avg_tokens_per_day):>15,}")
    print(f"{'Average cost / day:':<25} ${avg_cost_per_day:>14.2f}")
    print()
    print(f"{'Last 7 days tokens:':<25} {last_7_tokens:>15,}")
    print(f"{'Last 7 days cost:':<25} ${last_7_cost:>14.2f}")
    print(f"{'Last 7 days avg tokens:':<25} {int(last_7_tokens / 7):>15,}")
    print(f"{'Last 7 days avg cost:':<25} ${last_7_cost / 7:>14.2f}")
    print()
    print(f"{'Last 30 days tokens:':<25} {last_30_tokens:>15,}")
    print(f"{'Last 30 days cost:':<25} ${last_30_cost:>14.2f}")
    print(f"{'Last 30 days avg tokens:':<25} {int(last_30_tokens / 30):>15,}")
    print(f"{'Last 30 days avg cost:':<25} ${last_30_cost / 30:>14.2f}")

    if show_models and model_totals:
        print("\nSUMMARY BY MODEL")
        print("-" * 30)
        for model in sorted(model_totals.keys()):
            m_data = model_totals[model]
            days_active = len(m_data["days"])
            m_avg_tokens = (
                m_data["tokens"] / days_active if days_active else 0
            )
            m_avg_cost = m_data["cost"] / days_active if days_active else 0
            
            print(f"{model:<25}")
            print(f"  {'Total tokens:':<23} {m_data['tokens']:>15,}")
            print(f"  {'Total cost:':<23} ${m_data['cost']:>14.2f}")
            print(
                f"  {'Avg tokens / day:':<23} {int(m_avg_tokens):>15,}"
            )
            print(f"  {'Avg cost / day:':<23} ${m_avg_cost:>14.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Gemini token usage and costs."
    )
    parser.add_argument(
        "--model", action="store_true", help="Show breakdown per model."
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print only the raw total token count.",
    )

    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--today", action="store_true", help="Only show usage for today."
    )
    date_group.add_argument(
        "--yesterday",
        action="store_true",
        help="Only show usage for yesterday.",
    )
    date_group.add_argument(
        "--this-week",
        action="store_true",
        help="Usage for this week (from Monday).",
    )
    date_group.add_argument(
        "--last-week",
        action="store_true",
        help="Usage for last week (Mon-Sun).",
    )
    date_group.add_argument(
        "--this-month", action="store_true", help="Usage for this month."
    )
    date_group.add_argument(
        "--last-month", action="store_true", help="Usage for last month."
    )
    date_group.add_argument(
        "--date-range",
        help="Usage for a specific range (YYYY-MM-DD:YYYY-MM-DD).",
    )

    args = parser.parse_args()

    stats = aggregate_usage()

    if args.today:
        print_report(
            stats, show_models=args.model, today_only=True, raw=args.raw
        )
    else:
        start_date, end_date = None, None
        if args.yesterday:
            start_date, end_date = get_date_range("yesterday")
        elif args.this_week:
            start_date, end_date = get_date_range("this-week")
        elif args.last_week:
            start_date, end_date = get_date_range("last-week")
        elif args.this_month:
            start_date, end_date = get_date_range("this-month")
        elif args.last_month:
            start_date, end_date = get_date_range("last-month")
        elif args.date_range:
            start_date, end_date = get_date_range(args.date_range)

        if start_date and end_date:
            stats = filter_stats(stats, start_date, end_date)
            print_report(
                stats, show_models=args.model, today_only=False, raw=args.raw
            )
        else:
            print_report(
                stats, show_models=args.model, today_only=False, raw=args.raw
            )

    if not args.today and not args.raw and not any([
        args.yesterday, args.this_week, args.last_week,
        args.this_month, args.last_month, args.date_range
    ]):
        print_summary_statistics(stats, show_models=args.model)
