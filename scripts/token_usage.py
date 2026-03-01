#!/usr/bin/env python3
"""Calculates Gemini token usage and costs from session JSON files."""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ModelStats:
    """Statistics for a specific model usage."""

    sessions: Set[str] = field(default_factory=set)
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass
class PricingTier:
    """Rates for a specific pricing tier."""

    input_rate: float
    cached_rate: float
    output_rate: float


@dataclass
class ModelPricing:
    """Pricing configuration for a specific model or group of models."""

    small_context: PricingTier
    large_context: Optional[PricingTier] = None
    context_threshold: int = 200_000


@dataclass
class Config:
    """Pricing and model mapping configuration."""

    models: Dict[str, ModelPricing] = field(default_factory=dict)
    default_pricing: ModelPricing = field(
        default_factory=lambda: ModelPricing(
            small_context=PricingTier(2.00, 0.20, 12.00),
            large_context=PricingTier(4.00, 0.40, 18.00),
        )
    )

    def get_pricing(self, model_name: str) -> ModelPricing:
        """Finds the pricing for a given model name."""
        for pattern, pricing in self.models.items():
            if pattern in model_name.lower():
                return pricing
        return self.default_pricing


def load_config() -> Config:
    """Loads custom model mappings and rates from config files."""
    config = Config()

    # Pre-populate with known defaults
    config.models = {
        "gemini-3-pro-preview": ModelPricing(
            small_context=PricingTier(2.0, 0.2, 12.0),
            large_context=PricingTier(4.0, 0.4, 18.0),
        ),
        "gemini-3-flash-preview": ModelPricing(PricingTier(0.5, 0.05, 3.0)),
        "gemini-2.5-pro": ModelPricing(
            small_context=PricingTier(1.25, 0.125, 10.0),
            large_context=PricingTier(2.5, 0.25, 15.0),
        ),
        "gemini-2.5-flash-lite": ModelPricing(PricingTier(0.1, 0.01, 0.4)),
        "gemini-2.5-flash": ModelPricing(PricingTier(0.3, 0.03, 2.5)),
        "gemini-2.0-flash-lite": ModelPricing(PricingTier(0.075, 0.01, 0.3)),
        "gemini-2.0-flash": ModelPricing(PricingTier(0.1, 0.025, 0.4)),
        "gemini-1.5-pro": ModelPricing(PricingTier(3.5, 0.875, 10.5)),
        "gemini-1.5-flash": ModelPricing(PricingTier(0.075, 0.01875, 0.3)),
        "gemini-1.5-flash-8b": ModelPricing(PricingTier(0.0375, 0.01, 0.15)),
        "flash": ModelPricing(PricingTier(0.5, 0.05, 3.0)),
    }

    p = Path.home() / ".gemini" / "pricing.json"
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    return config

                # Handle custom model overrides
                custom_models = data.get("models", {})
                for pattern, pricing_data in custom_models.items():
                    if isinstance(pricing_data, list) and len(pricing_data) == 3:
                        # Simple PricingTier [in, cached, out]
                        tier = PricingTier(*[float(v) for v in pricing_data])
                        config.models[pattern.lower()] = ModelPricing(tier)
                    elif isinstance(pricing_data, dict):
                        # Tiered Pricing
                        small = pricing_data.get("small_context")
                        large = pricing_data.get("large_context")
                        threshold = pricing_data.get("context_threshold", 200_000)

                        if isinstance(small, list) and len(small) == 3:
                            s_tier = PricingTier(*[float(v) for v in small])
                            l_tier = None
                            if isinstance(large, list) and len(large) == 3:
                                l_tier = PricingTier(*[float(v) for v in large])

                            config.models[pattern.lower()] = ModelPricing(
                                small_context=s_tier,
                                large_context=l_tier,
                                context_threshold=threshold,
                            )
        except (json.JSONDecodeError, IOError, ValueError, TypeError):
            # Silently fallback to defaults if config is malformed
            pass
    return config


# Global configuration instance
CONFIG = load_config()


def reload_config() -> None:
    """Reloads the global configuration from disk."""
    global CONFIG
    CONFIG = load_config()


def calculate_cost(
    model: str, input_tokens: int, cached_tokens: int, output_tokens: int
) -> float:
    """Calculates cost based on model type and tiered pricing.

    Args:
        model: The model name.
        input_tokens: Total prompt tokens (including cached).
        cached_tokens: Tokens served from cache.
        output_tokens: Total output/thought tokens.
    """
    pricing = CONFIG.get_pricing(model)
    context_size = input_tokens

    tier = pricing.small_context
    if pricing.large_context and context_size > pricing.context_threshold:
        tier = pricing.large_context

    uncached_input = max(0, input_tokens - cached_tokens)

    return (
        uncached_input * tier.input_rate
        + cached_tokens * tier.cached_rate
        + output_tokens * tier.output_rate
    ) / 1_000_000


def aggregate_usage(
    base_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, ModelStats]]:
    """Aggregates Gemini token usage from session JSON files.

    Args:
        base_dir: Optional path to search for session files.
                 Defaults to ~/.gemini/tmp.

    Returns:
        A nested dictionary: stats[date][model] = ModelStats
    """
    if base_dir:
        tmp_dir = Path(base_dir)
        cache_file = tmp_dir / "usage_cache.json"
    else:
        gemini_dir = Path.home() / ".gemini"
        tmp_dir = gemini_dir / "tmp"
        cache_file = gemini_dir / "usage_cache.json"

    cache: Dict[str, Any] = {}
    if cache_file.exists():
        try:
            with cache_file.open("r", encoding="utf-8") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    stats: Dict[str, Dict[str, ModelStats]] = defaultdict(
        lambda: defaultdict(ModelStats)
    )

    if not tmp_dir.exists():
        return stats

    updated_cache: Dict[str, Any] = {}
    cache_dirty = False

    for session_file in tmp_dir.glob("**/session-*.json"):
        try:
            mtime = session_file.stat().st_mtime
            file_key = str(session_file)

            # Check cache for hits
            if file_key in cache and cache[file_key]["mtime"] == mtime:
                file_stats = cache[file_key]["stats"]
                updated_cache[file_key] = cache[file_key]

                for date_str, models in file_stats.items():
                    for model_name, s in models.items():
                        m_stats = stats[date_str][model_name]
                        m_stats.sessions.add(s["session_id"])

                        # In new cache version, s["input"] will be uncached input.
                        # In old cache version, it might be total input.
                        # If s["input"] > s["cached"] and we suspect it's total,
                        # we could subtract, but it's cleaner to just clear cache.
                        m_stats.input_tokens += s["input"]
                        m_stats.cached_tokens += s["cached"]
                        m_stats.output_tokens += s["output"]
                        m_stats.cost += s["cost"]
                continue

            # Cache miss or stale: parse the file
            cache_dirty = True
            with session_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                continue

            session_id = data.get("sessionId") or session_file.stem
            raw_start_time = data.get("startTime")
            start_time = str(raw_start_time) if raw_start_time else ""
            date_str = start_time.split("T")[0] if "T" in start_time else "unknown"

            # Temporary stats for this specific file to update cache
            file_record_stats: Dict[str, Dict[str, Any]] = defaultdict(
                lambda: defaultdict(dict)
            )

            messages = data.get("messages") or []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") == "gemini":
                    model_name = msg.get("model", "unknown")
                    tokens = msg.get("tokens") or {}

                    inp = tokens.get("input", 0)
                    cache_tokens = tokens.get("cached", 0)
                    out = tokens.get("output", 0) + tokens.get("thoughts", 0)

                    cost = calculate_cost(model_name, inp, cache_tokens, out)
                    uncached_inp = max(0, inp - cache_tokens)

                    # Update global aggregate
                    m_stats = stats[date_str][model_name]
                    m_stats.sessions.add(session_id)
                    m_stats.input_tokens += uncached_inp
                    m_stats.cached_tokens += cache_tokens
                    m_stats.output_tokens += out
                    m_stats.cost += cost

                    # Update file record for cache
                    r_stats = file_record_stats[date_str][model_name]
                    r_stats["session_id"] = session_id
                    r_stats["input"] = r_stats.get("input", 0) + uncached_inp
                    r_stats["cached"] = r_stats.get("cached", 0) + cache_tokens
                    r_stats["output"] = r_stats.get("output", 0) + out
                    r_stats["cost"] = r_stats.get("cost", 0) + cost

            updated_cache[file_key] = {"mtime": mtime, "stats": file_record_stats}

        except (json.JSONDecodeError, IOError, KeyError):
            continue

    if cache_dirty or len(updated_cache) != len(cache):
        try:
            with cache_file.open("w", encoding="utf-8") as f:
                json.dump(updated_cache, f)
        except IOError:
            pass
        except TypeError as e:
            # TypeError usually means something non-serializable got into the cache dict
            # We don't want to crash the whole tool, but we shouldn't silently ignore it during dev
            import sys

            print(f"Error: Failed to serialize cache: {e}", file=sys.stderr)

    return stats


def get_date_range(
    filter_name: str, today_obj: Optional[date] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Returns (start_date, end_date) strings for a given named filter.

    Args:
        filter_name: One of 'today', 'yesterday', 'this-week', 'last-week',
                     'this-month', 'last-month' or 'YYYY-MM-DD:YYYY-MM-DD'.
        today_obj: Optional date object for testing.

    Returns:
        A tuple of (start_date_string, end_date_string) or (None, None).
    """
    if today_obj is None:
        today_obj = datetime.now().date()

    if filter_name == "today":
        return today_obj.strftime("%Y-%m-%d"), today_obj.strftime("%Y-%m-%d")

    if filter_name == "yesterday":
        yesterday = today_obj - timedelta(days=1)
        return yesterday.strftime("%Y-%m-%d"), yesterday.strftime("%Y-%m-%d")

    if filter_name == "this-week":
        start = today_obj - timedelta(days=today_obj.weekday())
        return start.strftime("%Y-%m-%d"), today_obj.strftime("%Y-%m-%d")

    if filter_name == "last-week":
        end = today_obj - timedelta(days=today_obj.weekday() + 1)
        start = end - timedelta(days=6)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    if filter_name == "this-month":
        start = today_obj.replace(day=1)
        return start.strftime("%Y-%m-%d"), today_obj.strftime("%Y-%m-%d")

    if filter_name == "last-month":
        first_of_this_month = today_obj.replace(day=1)
        end = first_of_this_month - timedelta(days=1)
        start = end.replace(day=1)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    if ":" in filter_name:
        parts = filter_name.split(":")
        if len(parts) == 2:
            return parts[0], parts[1]

    return None, None


def filter_stats(
    stats: Dict[str, Dict[str, ModelStats]], start_date: str, end_date: str
) -> Dict[str, Dict[str, ModelStats]]:
    """Filters aggregated stats by date range.

    Args:
        stats: Aggregated stats dictionary.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).

    Returns:
        A filtered dictionary of stats.
    """
    filtered = {}
    for date_str, models in stats.items():
        if date_str == "unknown":
            continue
        if start_date <= date_str <= end_date:
            filtered[date_str] = models
    return filtered


def print_report(
    stats: Dict[str, Dict[str, ModelStats]],
    show_models: bool = False,
    today_only: bool = False,
    raw_tokens_only: bool = False,
) -> None:
    """Prints a formatted report of token usage.

    Args:
        stats: Aggregated stats dictionary.
        show_models: Whether to show per-model breakdown.
        today_only: Whether to limit the report to today's usage.
        raw_tokens_only: If True, only prints the total token count.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")

    if today_only:
        stats = {today_str: stats[today_str]} if today_str in stats else {}

    if not stats:
        if raw_tokens_only:
            print("0")
        else:
            print("No usage data found.")
        return

    grand_total_tokens = 0
    grand_total_cost = 0.0
    model_grand_totals: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"tokens": 0, "cost": 0.0}
    )

    if raw_tokens_only:
        for date_str in stats:
            for model_name in stats[date_str]:
                s = stats[date_str][model_name]
                grand_total_tokens += s.input_tokens + s.cached_tokens + s.output_tokens
        print(grand_total_tokens)
        return

    # Header configuration
    model_header = f"{'MODEL':<40} " if show_models else ""
    header = (
        f"{'DATE':<12} {model_header}{'SESS':<5} {'INPUT':>12} "
        f"{'CACHED':>12} {'OUTPUT':>12} {'TOTAL':>12} {'COST':>10}"
    )
    line_len = len(header) + 2
    print(header)
    print("-" * line_len)

    # Iteration by date
    for date_str in sorted(stats.keys()):
        display_date = f"{date_str}*" if date_str == today_str else f"{date_str:<11} "
        if not show_models:
            day_input = sum(s.input_tokens for s in stats[date_str].values())
            day_cached = sum(s.cached_tokens for s in stats[date_str].values())
            day_output = sum(s.output_tokens for s in stats[date_str].values())
            day_sessions: Set[str] = set()
            for s in stats[date_str].values():
                day_sessions.update(s.sessions)
            day_cost = sum(s.cost for s in stats[date_str].values())

            total = day_input + day_cached + day_output
            print(
                f"{display_date:<12} {len(day_sessions):<5} "
                f"{day_input:>12,} {day_cached:>12,} {day_output:>12,} "
                f"{total:>12,} ${day_cost:>8.2f}"
            )

            grand_total_tokens += total
            grand_total_cost += day_cost
        else:
            for model_name in sorted(stats[date_str].keys()):
                s = stats[date_str][model_name]
                total = s.input_tokens + s.cached_tokens + s.output_tokens

                print(
                    f"{display_date:<12} {model_name[:40]:<40} "
                    f"{len(s.sessions):<5} {s.input_tokens:>12,} "
                    f"{s.cached_tokens:>12,} {s.output_tokens:>12,} "
                    f"{total:>12,} ${s.cost:>8.2f}"
                )

                grand_total_tokens += total
                grand_total_cost += s.cost
                model_grand_totals[model_name]["tokens"] += total
                model_grand_totals[model_name]["cost"] += s.cost

    print("-" * line_len)
    if today_str in stats:
        print("* Note: Today's count is not complete (there can be more!)")

    if show_models and len(model_grand_totals) > 1:
        for model_name in sorted(model_grand_totals.keys()):
            m_stats = model_grand_totals[model_name]
            label = f"TOTALS ({model_name[:40]})"
            print(f"{label:<59} {m_stats['tokens']:>50,} ${m_stats['cost']:>8.2f}")
        print("-" * line_len)

    total_label = "TOTALS (ALL)" if show_models else "TOTALS"
    offset = 59 if show_models else 18
    print(
        f"{total_label:<{offset}} {grand_total_tokens:>50,} ${grand_total_cost:>8.2f}"
    )


def print_summary_statistics(
    stats: Dict[str, Dict[str, ModelStats]], show_models: bool = False
) -> None:
    """Prints aggregate summary statistics (averages, historical trends)."""
    if not stats:
        return

    today_obj = datetime.now().date()
    # Aggregate daily totals
    daily_token_totals: Dict[date, int] = defaultdict(int)
    daily_cost_totals: Dict[date, float] = defaultdict(float)
    model_totals: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"tokens": 0, "cost": 0.0, "days": set()}
    )

    for date_str, models in stats.items():
        if date_str == "unknown":
            continue
        try:
            d_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        for model_name, s in models.items():
            tokens = s.input_tokens + s.cached_tokens + s.output_tokens
            cost = s.cost

            daily_token_totals[d_obj] += tokens
            daily_cost_totals[d_obj] += cost

            model_totals[model_name]["tokens"] += tokens
            model_totals[model_name]["cost"] += cost
            model_totals[model_name]["days"].add(d_obj)

    if not daily_token_totals:
        return

    total_usage_days = len(daily_token_totals)
    grand_total_tokens = sum(daily_token_totals.values())
    grand_total_cost = sum(daily_cost_totals.values())

    print("\nSUMMARY STATISTICS (Averages per usage day)")
    print("-" * 30)

    gen_header = (
        f"{'PERIOD':<15} {'DAYS':>5} {'TOKENS':>15} {'COST':>12} "
        f"{'AVG TOKENS/D':>15} {'AVG COST/D':>12}"
    )
    print(gen_header)
    print("-" * len(gen_header))

    def print_period(label: str, usage_days: List[date]):
        d_count = len(usage_days)
        t_sum = sum(daily_token_totals[d] for d in usage_days)
        c_sum = sum(daily_cost_totals[d] for d in usage_days)
        t_avg = t_sum / d_count if d_count > 0 else 0
        c_avg = c_sum / d_count if d_count > 0 else 0
        print(
            f"{label:<15} {d_count:>5} {t_sum:>15,} ${c_sum:>10.2f} "
            f"{int(t_avg):>15,} ${c_avg:>10.2f}"
        )

    all_days = sorted(daily_token_totals.keys())
    print_period("All Time", all_days)

    last_7_cutoff = today_obj - timedelta(days=7)
    print_period("Last 7 Days", [d for d in all_days if d >= last_7_cutoff])

    last_30_cutoff = today_obj - timedelta(days=30)
    print_period("Last 30 Days", [d for d in all_days if d >= last_30_cutoff])

    if show_models and model_totals:
        print("\nSUMMARY BY MODEL")
        print("-" * 30)
        m_header = (
            f"{'MODEL':<45} {'DAYS':>5} {'TOTAL TOKENS':>15} "
            f"{'AVG TOKENS/D':>15} {'TOTAL COST':>12} {'AVG COST/D':>12}"
        )
        print(m_header)
        print("-" * len(m_header))

        for model_name in sorted(model_totals.keys()):
            m_data = model_totals[model_name]
            days_active = len(m_data["days"])
            m_avg_tokens = m_data["tokens"] / days_active if days_active else 0
            m_avg_cost = m_data["cost"] / days_active if days_active else 0

            print(
                f"{model_name:<45} {days_active:>5} "
                f"{m_data['tokens']:>15,} {int(m_avg_tokens):>15,} "
                f"${m_data['cost']:>10.2f} ${m_avg_cost:>10.2f}"
            )


def main() -> None:
    """CLI Entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate Gemini token usage and costs."
    )
    parser.add_argument(
        "--model", action="store_true", help="Show breakdown per model."
    )
    parser.add_argument(
        "--raw", action="store_true", help="Print only the raw total token count."
    )
    parser.add_argument(
        "dir",
        nargs="?",
        default=None,
        help="Optional path to search for session files.",
    )

    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--today", action="store_true", help="Only show usage for today."
    )
    date_group.add_argument(
        "--yesterday", action="store_true", help="Only show usage for yesterday."
    )
    date_group.add_argument(
        "--this-week", action="store_true", help="Usage for this week (from Monday)."
    )
    date_group.add_argument(
        "--last-week", action="store_true", help="Usage for last week (Mon-Sun)."
    )
    date_group.add_argument(
        "--this-month", action="store_true", help="Usage for this month."
    )
    date_group.add_argument(
        "--last-month", action="store_true", help="Usage for last month."
    )
    date_group.add_argument(
        "--date-range", help="Usage for a specific range (YYYY-MM-DD:YYYY-MM-DD)."
    )

    args = parser.parse_args()
    stats = aggregate_usage(args.dir)

    if args.today:
        print_report(
            stats, show_models=args.model, today_only=True, raw_tokens_only=args.raw
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
            stats, show_models=args.model, today_only=False, raw_tokens_only=args.raw
        )

    if (
        not args.today
        and not args.raw
        and not any(
            [
                args.yesterday,
                args.this_week,
                args.last_week,
                args.this_month,
                args.last_month,
                args.date_range,
            ]
        )
    ):
        print_summary_statistics(stats, show_models=args.model)


if __name__ == "__main__":
    main()
