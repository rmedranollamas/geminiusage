#!/usr/bin/env python3
"""Calculates Gemini token usage and costs from session JSON files."""

import argparse
import json
import os
import sys
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
    duration_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Returns total tokens (input + cached + output)."""
        return self.input_tokens + self.cached_tokens + self.output_tokens

    def add(self, other: "ModelStats") -> None:
        """Adds another ModelStats object to this one."""
        self.sessions.update(other.sessions)
        self.input_tokens += other.input_tokens
        self.cached_tokens += other.cached_tokens
        self.output_tokens += other.output_tokens
        self.cost += other.cost
        self.duration_seconds += other.duration_seconds


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


def format_duration(seconds: float) -> str:
    """Formats duration in seconds to a concise human-readable string (s, m, h)."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def format_duration_h(seconds: float) -> str:
    """Formats duration in seconds to a human-readable string (m, h) for summary stats."""
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def render_sparkline(values: List[int], width: int = 30) -> str:
    """Renders a simple ASCII sparkline from a list of values."""
    if not values:
        return ""

    # Ensure we have exactly 'width' elements
    if len(values) > width:
        values = values[-width:]
    elif len(values) < width:
        values = [0] * (width - len(values)) + values

    chars = " ▂▃▄▅▆▇█"
    max_val = max(values)
    if max_val == 0:
        return " " * width

    spark = ""
    for v in values:
        if v == 0:
            spark += " "
        else:
            # Ensure non-zero values get at least the first block character
            idx = int((v / max_val) * (len(chars) - 1))
            spark += chars[max(1, idx)]
    return spark


def discover_session_files(tmp_dir: Path) -> List[Path]:
    """Discovers Gemini session JSON files in the given directory.

    Args:
        tmp_dir: Path to search for session files.

    Returns:
        A list of Path objects for the discovered session files.
    """
    import os

    session_files = []
    try:
        # Most session files are in ~/.gemini/tmp/<uuid>/chats/session-*.json
        with os.scandir(str(tmp_dir)) as it:
            for entry in it:
                if entry.is_dir():
                    chats_path = os.path.join(entry.path, "chats")
                    if os.path.exists(chats_path):
                        with os.scandir(chats_path) as it_chats:
                            for f_entry in it_chats:
                                if (
                                    f_entry.is_file()
                                    and f_entry.name.startswith("session-")
                                    and f_entry.name.endswith(".json")
                                ):
                                    session_files.append(Path(f_entry.path))
                    else:
                        # Fallback for other structures
                        with os.scandir(entry.path) as it_uuid:
                            for f_entry in it_uuid:
                                if (
                                    f_entry.is_file()
                                    and f_entry.name.startswith("session-")
                                    and f_entry.name.endswith(".json")
                                ):
                                    session_files.append(Path(f_entry.path))
    except (IOError, OSError):
        pass

    # If no files found with targeted search, do a full walk as fallback
    if not session_files:
        for root, _, files in os.walk(str(tmp_dir)):
            for filename in files:
                if filename.startswith("session-") and filename.endswith(".json"):
                    session_files.append(Path(root) / filename)

    return session_files


def aggregate_usage(
    base_dir: Optional[Path] = None,
    since_mtime: Optional[float] = None,
    date_filter: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, ModelStats]]:
    """Aggregates Gemini token usage from session JSON files.

    Args:
        base_dir: Optional path to search for session files.
                 Defaults to ~/.gemini/tmp.
        since_mtime: Optional float timestamp. If provided, skips aggregating
                    stats for files modified before this time.
        date_filter: Optional set of date strings (YYYY-MM-DD). If provided,
                    only aggregates stats for these dates.

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

    cache_dirty = False
    # newly_parsed_entries stores only what we changed in this run
    newly_parsed_entries: Dict[str, Any] = {}

    session_files = discover_session_files(tmp_dir)

    for session_file in session_files:
        try:
            mtime = session_file.stat().st_mtime
            file_key = str(session_file)

            # Optimization: If we only care about recent files and this one is old
            if since_mtime and mtime < since_mtime:
                # Still check if we need to add its stats to the return value
                if file_key in cache and cache[file_key]["mtime"] == mtime:
                    file_stats = cache[file_key]["stats"]
                    for date_str, models in file_stats.items():
                        if date_filter and date_str not in date_filter:
                            continue
                        for model_name, s in models.items():
                            m_stats = stats[date_str][model_name]
                            m_stats.sessions.add(s["session_id"])
                            m_stats.input_tokens += s["input"]
                            m_stats.cached_tokens += s["cached"]
                            m_stats.output_tokens += s["output"]
                            m_stats.cost += s["cost"]
                            m_stats.duration_seconds += s.get("duration", 0.0)
                continue

            # Check cache for hits
            if file_key in cache and cache[file_key]["mtime"] == mtime:
                file_stats = cache[file_key]["stats"]
                for date_str, models in file_stats.items():
                    if date_filter and date_str not in date_filter:
                        continue
                    for model_name, s in models.items():
                        m_stats = stats[date_str][model_name]
                        m_stats.sessions.add(s["session_id"])
                        m_stats.input_tokens += s["input"]
                        m_stats.cached_tokens += s["cached"]
                        m_stats.output_tokens += s["output"]
                        m_stats.cost += s["cost"]
                        m_stats.duration_seconds += s.get("duration", 0.0)
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
            turn_start_ts = None
            turn_end_ts = None
            last_model_in_turn = "unknown"

            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                msg_type = msg.get("type")
                raw_ts = msg.get("timestamp")
                ts_val = None
                if raw_ts:
                    try:
                        ts_val = datetime.fromisoformat(
                            raw_ts.replace("Z", "+00:00")
                        ).timestamp()
                    except (ValueError, AttributeError):
                        pass

                if msg_type == "user" and ts_val:
                    if turn_start_ts and turn_end_ts:
                        stats[date_str][last_model_in_turn].duration_seconds += max(
                            0, turn_end_ts - turn_start_ts
                        )
                        file_record_stats[date_str][last_model_in_turn]["duration"] = (
                            file_record_stats[date_str][last_model_in_turn].get(
                                "duration", 0.0
                            )
                            + max(0, turn_end_ts - turn_start_ts)
                        )

                    turn_start_ts = ts_val
                    turn_end_ts = None
                    continue

                if msg_type != "gemini":
                    continue

                model_name = msg.get("model", "unknown")
                last_model_in_turn = model_name
                if ts_val:
                    turn_end_ts = ts_val

                tokens = msg.get("tokens") or {}

                # tool tokens included in input per clanker-stats analysis
                inp = tokens.get("input", 0) + tokens.get("tool", 0)
                cache_tokens = tokens.get("cached", 0)
                out = tokens.get("output", 0) + tokens.get("thoughts", 0)

                # Total fallback
                total_val = tokens.get("total", 0)
                if total_val > 0 and (inp + cache_tokens + out) == 0:
                    inp = total_val

                cost = calculate_cost(model_name, inp, cache_tokens, out)
                uncached_inp = max(0, inp - cache_tokens)

                # Update file record for cache (ALWAYS do this so cache is complete)
                r_stats = file_record_stats[date_str][model_name]
                r_stats["session_id"] = session_id
                r_stats["input"] = r_stats.get("input", 0) + uncached_inp
                r_stats["cached"] = r_stats.get("cached", 0) + cache_tokens
                r_stats["output"] = r_stats.get("output", 0) + out
                r_stats["cost"] = r_stats.get("cost", 0) + cost

                # Update global aggregate only if it passes the filters
                if date_filter and date_str not in date_filter:
                    continue

                m_stats = stats[date_str][model_name]
                m_stats.sessions.add(session_id)
                m_stats.input_tokens += uncached_inp
                m_stats.cached_tokens += cache_tokens
                m_stats.output_tokens += out
                m_stats.cost += cost

            if turn_start_ts and turn_end_ts:
                stats[date_str][last_model_in_turn].duration_seconds += max(
                    0, turn_end_ts - turn_start_ts
                )
                file_record_stats[date_str][last_model_in_turn]["duration"] = (
                    file_record_stats[date_str][last_model_in_turn].get("duration", 0.0)
                    + max(0, turn_end_ts - turn_start_ts)
                )

            newly_parsed_entries[file_key] = {
                "mtime": mtime,
                "stats": file_record_stats,
            }
        except (json.JSONDecodeError, IOError, KeyError):
            continue

    if cache_dirty:
        # Process-safe update using fcntl.flock
        lock_file_path = cache_file.with_suffix(".lock")
        try:
            import fcntl

            # 1. Acquire exclusive lock
            with lock_file_path.open("a+") as lock_f:
                fcntl.flock(lock_f, fcntl.LOCK_EX)

                # 2. Re-read the latest cache from disk to avoid stomping
                final_cache = {}
                if cache_file.exists():
                    try:
                        with cache_file.open("r", encoding="utf-8") as f:
                            final_cache = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        pass

                # 3. Merge ONLY the newly discovered entries
                final_cache.update(newly_parsed_entries)

                # 4. Atomic write using a temporary file
                import tempfile

                fd, temp_path = tempfile.mkstemp(
                    dir=str(cache_file.parent), prefix="usage_cache_"
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(final_cache, f)
                    os.replace(temp_path, str(cache_file))
                except (IOError, OSError, TypeError):
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise
        except (ImportError, IOError, OSError):
            # Fallback for systems without fcntl or permission issues
            # We still try to write even if locking fails, to be useful
            if newly_parsed_entries:
                try:
                    import tempfile

                    # Merge into our original cache and write
                    cache.update(newly_parsed_entries)
                    fd, temp_path = tempfile.mkstemp(
                        dir=str(cache_file.parent), prefix="usage_cache_"
                    )
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(cache, f)
                    os.replace(temp_path, str(cache_file))
                except (IOError, OSError, TypeError, NameError):
                    pass

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

    if filter_name.startswith("since:"):
        parts = filter_name.split(":")
        if len(parts) == 2:
            return parts[1], today_obj.strftime("%Y-%m-%d")

    if ":" in filter_name:
        parts = filter_name.split(":")
        if len(parts) == 2:
            return parts[0], parts[1]

    # Try to parse as a single date (meaning "since")
    try:
        datetime.strptime(filter_name, "%Y-%m-%d")
        if today_obj is None:
            today_obj = datetime.now().date()
        return filter_name, today_obj.strftime("%Y-%m-%d")
    except ValueError:
        pass

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
    show_hours: bool = False,
) -> None:
    """Prints a formatted report of token usage.

    Args:
        stats: Aggregated stats dictionary.
        show_models: Whether to show per-model breakdown.
        today_only: Whether to limit the report to today's usage.
        raw_tokens_only: If True, only prints the total token count.
        show_hours: If True, shows Focus Time instead of tokens.
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

    grand_total = ModelStats()
    model_grand_totals: Dict[str, ModelStats] = defaultdict(ModelStats)

    if raw_tokens_only:
        for date_stats in stats.values():
            for s in date_stats.values():
                grand_total.add(s)
        print(grand_total.total_tokens)
        return

    # Header configuration
    model_header = f"{'MODEL':<40} " if show_models else ""
    if show_hours:
        header = (
            f"{'DATE':<12} {model_header}{'SESS':<5} {'FOCUS TIME':>12} {'COST':>10}"
        )
    else:
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
            day_stats = ModelStats()
            for s in stats[date_str].values():
                day_stats.add(s)

            if show_hours:
                print(
                    f"{display_date:<12} {len(day_stats.sessions):<5} "
                    f"{format_duration(day_stats.duration_seconds):>12} ${day_stats.cost:>8.2f}"
                )
            else:
                print(
                    f"{display_date:<12} {len(day_stats.sessions):<5} "
                    f"{day_stats.input_tokens:>12,} {day_stats.cached_tokens:>12,} {day_stats.output_tokens:>12,} "
                    f"{day_stats.total_tokens:>12,} ${day_stats.cost:>8.2f}"
                )

            grand_total.add(day_stats)
        else:
            for model_name in sorted(stats[date_str].keys()):
                s = stats[date_str][model_name]
                if show_hours:
                    print(
                        f"{display_date:<12} {model_name[:40]:<40} "
                        f"{len(s.sessions):<5} {format_duration(s.duration_seconds):>12} ${s.cost:>8.2f}"
                    )
                else:
                    print(
                        f"{display_date:<12} {model_name[:40]:<40} "
                        f"{len(s.sessions):<5} {s.input_tokens:>12,} "
                        f"{s.cached_tokens:>12,} {s.output_tokens:>12,} "
                        f"{s.total_tokens:>12,} ${s.cost:>8.2f}"
                    )

                grand_total.add(s)
                model_grand_totals[model_name].add(s)

    print("-" * line_len)
    if today_str in stats:
        print("* Note: Today's count is not complete (there can be more!)")

    if show_models and len(model_grand_totals) > 1:
        for model_name in sorted(model_grand_totals.keys()):
            m_stats = model_grand_totals[model_name]
            label = f"TOTALS ({model_name[:40]})"
            if show_hours:
                print(
                    f"{label:<59} {format_duration(m_stats.duration_seconds):>50} ${m_stats.cost:>8.2f}"
                )
            else:
                print(f"{label:<59} {m_stats.total_tokens:>50,} ${m_stats.cost:>8.2f}")
        print("-" * line_len)

    total_label = "TOTALS (ALL)" if show_models else "TOTALS"
    offset = 59 if show_models else 18
    if show_hours:
        print(
            f"{total_label:<{offset}} {format_duration(grand_total.duration_seconds):>50} ${grand_total.cost:>8.2f}"
        )
    else:
        print(
            f"{total_label:<{offset}} {grand_total.total_tokens:>50,} ${grand_total.cost:>8.2f}"
        )


def print_summary_statistics(
    stats: Dict[str, Dict[str, ModelStats]],
    show_models: bool = False,
    show_hours: bool = False,
    since_date: Optional[str] = None,
) -> None:
    """Prints aggregate summary statistics (averages, historical trends)."""
    if not stats:
        return

    today_obj = datetime.now().date()
    # Aggregate daily totals
    daily_totals: Dict[date, ModelStats] = defaultdict(ModelStats)
    model_totals: Dict[str, ModelStats] = defaultdict(ModelStats)
    model_days: Dict[str, Set[date]] = defaultdict(set)

    for date_str, models in stats.items():
        if date_str == "unknown":
            continue
        try:
            d_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        for model_name, s in models.items():
            daily_totals[d_obj].add(s)
            model_totals[model_name].add(s)
            model_days[model_name].add(d_obj)

    if not daily_totals:
        return

    all_days = sorted(daily_totals.keys())

    if not since_date:
        # Sparkline (last 30 days)
        spark_values = []
        start_date = today_obj - timedelta(days=29)
        for i in range(30):
            d = start_date + timedelta(days=i)
            day_stat = daily_totals.get(d, ModelStats())
            val = (
                int(day_stat.duration_seconds) if show_hours else day_stat.total_tokens
            )
            spark_values.append(val)

        print(f"\nTREND (Last 30 days {'Focus' if show_hours else 'Tokens'})")
        print(f"{render_sparkline(spark_values)}")

    print("\nSUMMARY STATISTICS (Averages per usage day)")
    print("-" * 30)

    gen_header = (
        f"{'PERIOD':<16} {'DAYS':>5} {'TOKENS':>15} {'FOCUS':>10} {'COST':>12} "
        f"{'AVG TOKENS/D':>15} {'AVG COST/D':>12}"
    )
    print(gen_header)
    print("-" * len(gen_header))

    def print_period(label: str, usage_days: List[date]):
        d_count = len(usage_days)
        period_stats = ModelStats()
        for d in usage_days:
            period_stats.add(daily_totals[d])

        t_avg = period_stats.total_tokens / d_count if d_count > 0 else 0
        c_avg = period_stats.cost / d_count if d_count > 0 else 0
        print(
            f"{label[:16]:<16} {d_count:>5} {period_stats.total_tokens:>15,} {format_duration_h(period_stats.duration_seconds):>10} ${period_stats.cost:>10.2f} "
            f"{int(t_avg):>15,} ${c_avg:>10.2f}"
        )

    if since_date:
        print_period(f"Since {since_date}", all_days)
    else:
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
            f"{'FOCUS':>10} {'AVG TOKENS/D':>15} {'TOTAL COST':>12} {'AVG COST/D':>12}"
        )
        print(m_header)
        print("-" * len(m_header))

        for model_name in sorted(model_totals.keys()):
            m_stats = model_totals[model_name]
            days_active = len(model_days[model_name])
            m_avg_tokens = m_stats.total_tokens / days_active if days_active else 0
            m_avg_cost = m_stats.cost / days_active if days_active else 0

            print(
                f"{model_name:<45} {days_active:>5} "
                f"{m_stats.total_tokens:>15,} {format_duration(m_stats.duration_seconds):>10} "
                f"{int(m_avg_tokens):>15,} ${m_stats.cost:>10.2f} ${m_avg_cost:>10.2f}"
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
        "--hours",
        action="store_true",
        help="Show Focus Time (active hours) instead of tokens.",
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
    date_group.add_argument(
        "--since", help="Usage since a specific date (YYYY-MM-DD), inclusive."
    )

    args = parser.parse_args()

    # Determine filter name and calculate date range
    filter_name = "all"
    if args.today:
        filter_name = "today"
    elif args.yesterday:
        filter_name = "yesterday"
    elif args.this_week:
        filter_name = "this-week"
    elif args.last_week:
        filter_name = "last-week"
    elif args.this_month:
        filter_name = "this-month"
    elif args.last_month:
        filter_name = "last-month"
    elif args.date_range:
        filter_name = args.date_range
    elif args.since:
        filter_name = f"since:{args.since}"

    start_date, end_date = get_date_range(filter_name)

    # Optimization: Calculate since_mtime and date_filter to speed up aggregation
    since_mtime = None
    date_filter = None

    if start_date and end_date:
        try:
            start_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
            date_filter = {
                (start_obj + timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range((end_obj - start_obj).days + 1)
            }
            since_mtime = datetime.combine(start_obj, datetime.min.time()).timestamp()
        except ValueError:
            if filter_name.startswith("since:"):
                print(
                    f"Error: Invalid date format for --since: {args.since}. Use YYYY-MM-DD."
                )
                sys.exit(1)

    stats = aggregate_usage(args.dir, since_mtime=since_mtime, date_filter=date_filter)

    if args.today:
        print_report(
            stats,
            show_models=args.model,
            today_only=True,
            raw_tokens_only=args.raw,
            show_hours=args.hours,
        )
    else:
        if start_date and end_date:
            stats = filter_stats(stats, start_date, end_date)

        print_report(
            stats,
            show_models=args.model,
            today_only=False,
            raw_tokens_only=args.raw,
            show_hours=args.hours,
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
                args.since,
            ]
        )
    ):
        print_summary_statistics(
            stats, show_models=args.model, show_hours=args.hours, since_date=args.since
        )


if __name__ == "__main__":
    main()
