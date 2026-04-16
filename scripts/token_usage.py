#!/usr/bin/env python3
"""Calculates Gemini token usage and costs from session JSON files."""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add scripts directory to path to allow importing local modules
sys.path.append(os.path.dirname(__file__))

# Optional Antigravity status module
antigravity_status: Any = None
try:
    import antigravity_status as ag_status

    antigravity_status = ag_status
except ImportError:
    pass


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
    """Calculates cost based on model type and tiered pricing."""
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


def discover_session_files(
    scan_dirs: List[Path],
    since_mtime: Optional[float] = None,
) -> List[Path]:
    """Discovers Gemini session JSON files in the given directories."""
    session_files = []

    for tmp_dir in scan_dirs:
        if not tmp_dir.exists():
            continue

        dir_files = []
        try:
            with os.scandir(str(tmp_dir)) as it:
                for entry in it:
                    if entry.is_dir():
                        try:
                            chats_path = os.path.join(entry.path, "chats")
                            if os.path.exists(chats_path):
                                with os.scandir(chats_path) as it_chats:
                                    for f_entry in it_chats:
                                        if (
                                            f_entry.is_file()
                                            and f_entry.name.startswith("session-")
                                            and (
                                                f_entry.name.endswith(".json")
                                                or f_entry.name.endswith(".jsonl")
                                            )
                                        ):
                                            if (
                                                not since_mtime
                                                or f_entry.stat().st_mtime
                                                >= since_mtime
                                            ):
                                                dir_files.append(Path(f_entry.path))
                            else:
                                with os.scandir(entry.path) as it_uuid:
                                    for f_entry in it_uuid:
                                        if (
                                            f_entry.is_file()
                                            and f_entry.name.startswith("session-")
                                            and (
                                                f_entry.name.endswith(".json")
                                                or f_entry.name.endswith(".jsonl")
                                            )
                                        ):
                                            if (
                                                not since_mtime
                                                or f_entry.stat().st_mtime
                                                >= since_mtime
                                            ):
                                                dir_files.append(Path(f_entry.path))
                        except (IOError, OSError):
                            continue
        except (IOError, OSError):
            pass

        if not dir_files:
            for root, _, files in os.walk(str(tmp_dir)):
                for filename in files:
                    if filename.startswith("session-") and (
                        filename.endswith(".json") or filename.endswith(".jsonl")
                    ):
                        try:
                            f_path = Path(root) / filename
                            if not since_mtime or f_path.stat().st_mtime >= since_mtime:
                                dir_files.append(f_path)
                        except (IOError, OSError):
                            continue

        session_files.extend(dir_files)

    return session_files


def aggregate_usage(
    base_dir: Optional[Path] = None,
    since_mtime: Optional[float] = None,
    since_timestamp: Optional[float] = None,
    date_filter: Optional[Set[str]] = None,
    force_refresh: bool = False,
    fast_fail: bool = False,
) -> Dict[str, Dict[str, ModelStats]]:
    """Aggregates Gemini token usage from session JSON files."""
    if base_dir:
        scan_dirs = [Path(base_dir)]
        cache_file = Path(base_dir) / "usage_cache.json"
    else:
        gemini_dir = Path.home() / ".gemini"
        scan_dirs = [gemini_dir / "tmp", gemini_dir / "history"]
        cache_file = gemini_dir / "usage_cache.json"

    lock_file_path = cache_file.with_suffix(".lock")
    stats: Dict[str, Dict[str, ModelStats]] = defaultdict(
        lambda: defaultdict(ModelStats)
    )

    if not any(d.exists() for d in scan_dirs):
        return stats

    def perform_aggregation(
        cache: Dict[str, Any],
    ) -> Tuple[Dict[str, Dict[str, ModelStats]], Dict[str, Any], bool]:
        agg_stats: Dict[str, Dict[str, ModelStats]] = defaultdict(
            lambda: defaultdict(ModelStats)
        )
        newly_parsed: Dict[str, Any] = {}
        dirty = force_refresh

        times = [t for t in [since_mtime, since_timestamp] if t is not None]
        discover_since = (min(times) - 3600) if times else None

        session_files = discover_session_files(scan_dirs, since_mtime=discover_since)
        session_file_keys = {str(f) for f in session_files}

        for session_file in session_files:
            try:
                stat = session_file.stat()
                mtime = stat.st_mtime
                size = stat.st_size
                file_key = str(session_file)

                if (
                    not force_refresh
                    and file_key in cache
                    and cache[file_key].get("mtime") == mtime
                    and cache[file_key].get("size") == size
                ):
                    continue

                if session_file.suffix == ".jsonl":
                    messages = []
                    session_id = session_file.stem
                    session_date_str = "unknown"
                    with session_file.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            if "$set" in obj:
                                continue

                            if "sessionId" in obj:
                                session_id = obj.get("sessionId") or session_id
                                raw_start_time = obj.get("startTime")
                                start_time = (
                                    str(raw_start_time) if raw_start_time else ""
                                )
                                session_date_str = (
                                    start_time.split("T")[0]
                                    if "T" in start_time
                                    else "unknown"
                                )
                            elif "type" in obj:
                                messages.append(obj)
                else:
                    with session_file.open("r", encoding="utf-8") as f:
                        data = json.load(f)

                    if not isinstance(data, dict):
                        continue

                    session_id = data.get("sessionId") or session_file.stem
                    raw_start_time = data.get("startTime")
                    start_time = str(raw_start_time) if raw_start_time else ""
                    session_date_str = (
                        start_time.split("T")[0] if "T" in start_time else "unknown"
                    )
                    messages = data.get("messages") or []

                dirty = True

                file_record_stats: Dict[str, Dict[str, Any]] = defaultdict(
                    lambda: defaultdict(dict)
                )
                turn_start_ts = None
                turn_end_ts = None
                last_model_in_turn = "unknown"
                last_turn_bucket = session_date_str

                for msg in messages:
                    if not isinstance(msg, dict):
                        continue

                    msg_type = msg.get("type")
                    raw_ts = msg.get("timestamp")
                    ts_val = None
                    msg_bucket = session_date_str

                    if raw_ts:
                        try:
                            dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                            ts_val = dt.timestamp()
                            # Use hourly buckets for higher precision caching if timestamp is available
                            msg_bucket = dt.strftime("%Y-%m-%dT%H")
                        except (ValueError, AttributeError):
                            pass

                    # Filter by precise timestamp if requested
                    if since_timestamp and ts_val and ts_val < since_timestamp:
                        continue

                    if msg_type == "user" and ts_val:
                        if turn_start_ts and turn_end_ts:
                            file_record_stats[last_turn_bucket][last_model_in_turn][
                                "duration"
                            ] = file_record_stats[last_turn_bucket][
                                last_model_in_turn
                            ].get("duration", 0.0) + max(0, turn_end_ts - turn_start_ts)

                        turn_start_ts = ts_val
                        turn_end_ts = None
                        last_turn_bucket = msg_bucket
                        continue

                    if msg_type != "gemini":
                        continue

                    model_name = msg.get("model", "unknown")
                    last_model_in_turn = model_name
                    if ts_val:
                        turn_end_ts = ts_val

                    tokens = msg.get("tokens") or {}
                    inp = tokens.get("input", 0) + tokens.get("tool", 0)
                    cache_tokens = tokens.get("cached", 0)
                    out = tokens.get("output", 0) + tokens.get("thoughts", 0)

                    total_val = tokens.get("total", 0)
                    if total_val > 0 and (inp + cache_tokens + out) == 0:
                        inp = total_val

                    cost = calculate_cost(model_name, inp, cache_tokens, out)
                    uncached_inp = max(0, inp - cache_tokens)

                    r_stats = file_record_stats[msg_bucket][model_name]
                    r_stats["session_id"] = session_id
                    r_stats["input"] = r_stats.get("input", 0) + uncached_inp
                    r_stats["cached"] = r_stats.get("cached", 0) + cache_tokens
                    r_stats["output"] = r_stats.get("output", 0) + out
                    r_stats["cost"] = r_stats.get("cost", 0) + cost

                if turn_start_ts and turn_end_ts:
                    file_record_stats[last_turn_bucket][last_model_in_turn][
                        "duration"
                    ] = file_record_stats[last_turn_bucket][last_model_in_turn].get(
                        "duration", 0.0
                    ) + max(0, turn_end_ts - turn_start_ts)

                newly_parsed[file_key] = {
                    "mtime": mtime,
                    "size": size,
                    "stats": file_record_stats,
                }
            except (json.JSONDecodeError, IOError, KeyError):
                continue

        cache.update(newly_parsed)

        for file_key, cached_data in cache.items():
            if force_refresh and file_key not in session_file_keys:
                continue

            file_stats = cached_data.get("stats", {})
            for bucket_str, models in file_stats.items():
                # Apply date filter (bucket_str could be YYYY-MM-DD or YYYY-MM-DDTHH)
                date_str = bucket_str[:10]
                if date_filter and date_str not in date_filter:
                    continue

                # Apply precise timestamp filter if available in bucket string
                if since_timestamp and "T" in bucket_str:
                    try:
                        bucket_ts = datetime.strptime(
                            bucket_str, "%Y-%m-%dT%H"
                        ).timestamp()
                        # If the bucket is entirely before our cutoff, skip it.
                        # Note: a bucket for hour 14 covers [14:00, 15:00).
                        # If since_timestamp is 14:30, we should ideally include
                        # messages from that bucket, but we can't tell which ones
                        # from the aggregated bucket. However, since we re-parse
                        # files that are new/changed, this mostly affects older
                        # cached entries. For 24h rolling, it's "close enough"
                        # to filter at hourly granularity for cached data.
                        if bucket_ts + 3600 < since_timestamp:
                            continue
                    except ValueError:
                        pass

                for model_name, s in models.items():
                    # Normalize back to daily for the final report
                    m_stats = agg_stats[date_str][model_name]
                    if "session_id" in s:
                        m_stats.sessions.add(s["session_id"])
                    m_stats.input_tokens += s.get("input", 0)
                    m_stats.cached_tokens += s.get("cached", 0)
                    m_stats.output_tokens += s.get("output", 0)
                    m_stats.cost += s.get("cost", 0.0)
                    m_stats.duration_seconds += s.get("duration", 0.0)

        return agg_stats, newly_parsed, dirty

    can_write_cache = True
    stats: Any = None
    try:
        import fcntl

        with lock_file_path.open("a+") as lock_f:
            try:
                # Always use non-blocking lock to adhere to mandate in GEMINI.md
                fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                has_lock = True
            except BlockingIOError:
                has_lock = False
                if fast_fail:
                    sys.exit(2)
            except (IOError, OSError):
                has_lock = False
                can_write_cache = False

            current_cache = {}
            if cache_file.exists():
                try:
                    with cache_file.open("r", encoding="utf-8") as f:
                        current_cache = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass

            stats, newly_parsed_entries, cache_dirty = perform_aggregation(
                current_cache
            )

            if cache_dirty and has_lock:
                current_cache.update(newly_parsed_entries)
                if force_refresh:
                    session_files = discover_session_files(scan_dirs)
                    disk_keys = {str(f) for f in session_files}
                    current_cache = {
                        k: v for k, v in current_cache.items() if k in disk_keys
                    }

                import tempfile

                fd, temp_path = tempfile.mkstemp(
                    dir=str(cache_file.parent), prefix="usage_cache_"
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(current_cache, f)
                    os.replace(temp_path, str(cache_file))
                except (IOError, OSError, TypeError):
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise
    except PermissionError as e:
        print(
            f"Warning: Permission denied for lock file {lock_file_path}: {e}. Cache writing disabled.",
            file=sys.stderr,
        )
        can_write_cache = False
        if fast_fail:
            sys.exit(2)
    except (ImportError, IOError, OSError):
        can_write_cache = False
        if fast_fail and stats is None:
            sys.exit(2)

    if stats is None:
        current_cache = {}
        if cache_file.exists() and not force_refresh:
            try:
                with cache_file.open("r", encoding="utf-8") as f:
                    current_cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        stats, newly_parsed_entries, cache_dirty = perform_aggregation(current_cache)
        if cache_dirty and can_write_cache:
            try:
                import tempfile

                current_cache.update(newly_parsed_entries)
                fd, temp_path = tempfile.mkstemp(
                    dir=str(cache_file.parent), prefix="usage_cache_"
                )
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(current_cache, f)
                os.replace(temp_path, str(cache_file))
            except (IOError, OSError, TypeError, NameError):
                pass

    return stats


def get_date_range(
    filter_name: str,
    today_obj: Optional[date] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Returns (start_date, end_date) strings for a given named filter."""
    if today_obj is None:
        today_obj = datetime.now().date()

    if filter_name == "all":
        return None, None

    if filter_name == "24h":
        start = today_obj - timedelta(days=1)
        return start.strftime("%Y-%m-%d"), today_obj.strftime("%Y-%m-%d")

    if filter_name == "today":
        d_str = today_obj.strftime("%Y-%m-%d")
        return d_str, d_str

    if filter_name == "yesterday":
        d_str = (today_obj - timedelta(days=1)).strftime("%Y-%m-%d")
        return d_str, d_str

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

    try:
        datetime.strptime(filter_name, "%Y-%m-%d")
        return filter_name, today_obj.strftime("%Y-%m-%d")
    except ValueError:
        pass

    return None, None


def filter_stats(
    stats: Dict[str, Dict[str, ModelStats]], start_date: str, end_date: str
) -> Dict[str, Dict[str, ModelStats]]:
    """Filters aggregated stats by date range."""
    filtered = {}
    for date_str, models in stats.items():
        if date_str == "unknown":
            continue
        if start_date <= date_str <= end_date:
            filtered[date_str] = models
    return filtered


def get_antigravity_summary() -> str:
    """Returns a compact Antigravity status summary."""
    if not antigravity_status:
        return ""

    try:
        status = antigravity_status.get_status()
        if not status or not status.get("running"):
            return ""

        if not status.get("connected"):
            return " | AGY: DISC"

        models = status.get("models", [])
        if not models:
            return " | AGY: OK"

        low_model = models[0]
        rem_pct = int(low_model["remaining"] * 100)
        return f" | AGY: {low_model['label']} {rem_pct}%"
    except Exception:
        return ""


def print_report(
    stats: Dict[str, Dict[str, ModelStats]],
    show_models: bool = False,
    today_only: bool = False,
    raw_tokens_only: bool = False,
    show_hours: bool = False,
    show_antigravity: bool = False,
) -> None:
    """Prints a formatted report of token usage."""
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if today_only:
        stats = {today_str: stats[today_str]} if today_str in stats else {}

    grand_total = ModelStats()
    if stats:
        for date_stats in stats.values():
            for s in date_stats.values():
                grand_total.add(s)

    if raw_tokens_only:
        output = str(grand_total.total_tokens)
        if show_antigravity:
            output += get_antigravity_summary()
        print(output)
        return

    if not stats:
        print("No usage data found.")
        return

    model_grand_totals: Dict[str, ModelStats] = defaultdict(ModelStats)

    model_header = f"{'MODEL':<40} " if show_models else ""
    if show_hours:
        header = (
            f"{'DATE':<12} {model_header}{'SESS':<5} {'ACTIVE TIME':>12} {'COST':>10}"
        )
    else:
        header = (
            f"{'DATE':<12} {model_header}{'SESS':<5} {'INPUT':>12} "
            f"{'CACHED':>12} {'OUTPUT':>12} {'TOTAL':>12} {'COST':>10}"
        )
    line_len = len(header) + 2
    print(header)
    print("-" * line_len)

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

    if show_antigravity:
        agy_summary = get_antigravity_summary()
        if agy_summary:
            print(f"\nAntigravity Status:{agy_summary.replace(' | AGY:', '')}")


def print_summary_statistics(
    stats: Dict[str, Dict[str, ModelStats]],
    show_models: bool = False,
    show_hours: bool = False,
    since_date: Optional[str] = None,
) -> None:
    """Prints aggregate summary statistics (averages, historical trends)."""
    if not stats:
        return

    today_obj = datetime.now(timezone.utc).date()
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

    avg_days = [d for d in daily_totals.keys() if d < today_obj]
    if not avg_days:
        avg_days = list(daily_totals.keys())

    num_days = len(avg_days)
    total_stats = ModelStats()
    for d in avg_days:
        total_stats.add(daily_totals[d])

    avg_tokens = total_stats.total_tokens / num_days if num_days else 0
    avg_cost = total_stats.cost / num_days if num_days else 0

    print("\nSUMMARY STATISTICS")
    if since_date:
        print(f"Period: Since {since_date} ({len(daily_totals)} days active)")
    else:
        print(f"Period: All Time ({len(daily_totals)} days active)")

    print("-" * 40)
    if show_hours:
        avg_hours = total_stats.duration_seconds / num_days if num_days else 0
        print(f"Daily Average Active Time: {format_duration_h(avg_hours)}")
    else:
        print(f"Daily Average Tokens:      {int(avg_tokens):,}")
    print(f"Daily Average Cost:        ${avg_cost:.2f}")

    if show_models:
        print("\nMODEL BREAKDOWN")
        print(
            f"{'MODEL':<45} {'DAYS':>5} {'TOTAL':>15} {'ACTIVE':>10} {'AVG TOK':>15} {'COST':>10} {'AVG COST':>10}"
        )
        print("-" * 115)
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
        "--antigravity",
        "--agy",
        action="store_true",
        help="Include Antigravity status summary.",
    )
    parser.add_argument(
        "--raw", action="store_true", help="Print only the raw total token count."
    )
    parser.add_argument(
        "--hours",
        action="store_true",
        help="Show Active Time (active hours) instead of tokens.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-scan of all sessions and update cache.",
    )
    parser.add_argument(
        "--fast-fail",
        action="store_true",
        help="Exit immediately if cache lock is held by another process.",
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
        "--rolling-24h", "--24h", action="store_true", dest="rolling_24h", help="Show usage for the last 24 hours."
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

    filter_name = "all"
    if args.today:
        filter_name = "today"
    elif args.rolling_24h:
        filter_name = "24h"
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

    since_mtime = None
    since_timestamp = None
    date_filter = None

    if filter_name == "24h":
        import time

        since_timestamp = time.time() - 86400

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
                    f"Error: Invalid date format for --since: {args.since}. Use YYYY-MM-DD.",
                    file=sys.stderr,
                )
                sys.exit(1)

    stats = aggregate_usage(
        args.dir,
        since_mtime=since_mtime,
        since_timestamp=since_timestamp,
        date_filter=date_filter,
        force_refresh=args.refresh,
        fast_fail=args.fast_fail,
    )

    if args.today or args.rolling_24h:
        print_report(
            stats,
            show_models=args.model,
            today_only=args.today,
            raw_tokens_only=args.raw,
            show_hours=args.hours,
            show_antigravity=args.antigravity,
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
            show_antigravity=args.antigravity,
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
