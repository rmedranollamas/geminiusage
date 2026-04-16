#!/usr/bin/env python3
"""Tests for token_usage.py logic."""

import io
import json
import os
import sys
import time
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import token_usage


class TestTokenUsage(unittest.TestCase):
    """Core logic tests for token usage aggregation and calculation."""

    def test_aggregation_with_caching(self) -> None:
        """Verifies that aggregation works correctly and respects caching."""
        with TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            # Create a mock session file structure
            chat_dir = tmp_path / "project1" / "chats"
            chat_dir.mkdir(parents=True)

            session_data = {
                "sessionId": "test-session",
                "startTime": "2026-01-20T12:00:00Z",
                "messages": [
                    {
                        "type": "gemini",
                        "model": "gemini-3-flash",
                        "tokens": {
                            "input": 100,
                            "cached": 50,
                            "output": 20,
                            "thoughts": 10,
                        },
                    },
                ],
            }

            session_file = chat_dir / "session-1.json"
            with session_file.open("w") as f:
                json.dump(session_data, f)

            # First run: should parse file
            stats = token_usage.aggregate_usage(base_dir=tmp_path)
            self.assertIn("2026-01-20", stats)
            model_stats = stats["2026-01-20"]["gemini-3-flash"]
            self.assertEqual(model_stats.input_tokens, 50)  # 100 total - 50 cached = 50
            self.assertEqual(model_stats.output_tokens, 30)  # 20 + 10
            self.assertEqual(model_stats.cached_tokens, 50)

            # Verify cache file was created
            cache_file = tmp_path / "usage_cache.json"
            self.assertTrue(cache_file.exists())

            # Second run: should use cache
            stats_cached = token_usage.aggregate_usage(base_dir=tmp_path)
            self.assertEqual(
                stats_cached["2026-01-20"]["gemini-3-flash"].input_tokens, 50
            )
            self.assertEqual(
                stats_cached["2026-01-20"]["gemini-3-flash"].cached_tokens, 50
            )

    @patch("fcntl.flock")
    def test_aggregation_non_blocking_lock(self, mock_flock) -> None:
        """Verifies that if the lock cannot be acquired, we still return stats but do not write cache."""
        mock_flock.side_effect = BlockingIOError("Lock busy")
        with TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            chat_dir = tmp_path / "project1" / "chats"
            chat_dir.mkdir(parents=True)

            session_data = {
                "sessionId": "test-session-concurrent",
                "startTime": "2026-01-20T12:00:00Z",
                "messages": [
                    {
                        "type": "gemini",
                        "model": "gemini-3-flash",
                        "tokens": {"input": 100},
                    },
                ],
            }
            session_file = chat_dir / "session-1.json"
            with session_file.open("w") as f:
                json.dump(session_data, f)

            # Execution with simulated lock contention
            stats = token_usage.aggregate_usage(base_dir=tmp_path)

            # Stats should still be parsed from disk correctly
            self.assertIn("2026-01-20", stats)
            self.assertEqual(stats["2026-01-20"]["gemini-3-flash"].input_tokens, 100)

            # But the cache file should NOT have been created/written
            cache_file = tmp_path / "usage_cache.json"
            self.assertFalse(cache_file.exists())

    def test_aggregation_permission_error_on_lock(self) -> None:
        """Verifies that PermissionError on lock file disables cache writing safely."""
        with TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            chat_dir = tmp_path / "project1" / "chats"
            chat_dir.mkdir(parents=True)

            session_data = {
                "sessionId": "test-session-permission",
                "startTime": "2026-01-20T12:00:00Z",
                "messages": [
                    {
                        "type": "gemini",
                        "model": "gemini-3-flash",
                        "tokens": {"input": 100},
                    },
                ],
            }
            session_file = chat_dir / "session-1.json"
            with session_file.open("w") as f:
                json.dump(session_data, f)

            original_open = Path.open

            def mock_open_func(self_obj, *args, **kwargs):
                if self_obj.name.endswith(".lock"):
                    raise PermissionError("Permission denied")
                return original_open(self_obj, *args, **kwargs)

            with patch("pathlib.Path.open", autospec=True, side_effect=mock_open_func):
                stats = token_usage.aggregate_usage(base_dir=tmp_path)

            # Stats should still be parsed from disk correctly
            self.assertIn("2026-01-20", stats)
            self.assertEqual(stats["2026-01-20"]["gemini-3-flash"].input_tokens, 100)

            # But the cache file should NOT have been created/written
            cache_file = tmp_path / "usage_cache.json"
            self.assertFalse(cache_file.exists())

    def test_aggregation_robustness(self) -> None:
        """Verifies that aggregation handles malformed or null-valued JSON fields."""
        with TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            chat_dir = tmp_path / "chats"
            chat_dir.mkdir()

            # Session with null tokens and null messages
            session_nulls = {
                "sessionId": "null-session",
                "startTime": "2026-01-21T12:00:00Z",
                "messages": [
                    {"type": "gemini", "model": "gemini-3-flash", "tokens": None},
                    None,  # Malformed message
                ],
            }

            with (chat_dir / "session-nulls.json").open("w") as f:
                json.dump(session_nulls, f)

            # This should not crash
            stats = token_usage.aggregate_usage(base_dir=tmp_path)
            self.assertIn("2026-01-21", stats)
            model_stats = stats["2026-01-21"]["gemini-3-flash"]
            self.assertEqual(model_stats.input_tokens, 0)

    def test_calculate_cost_tiers(self) -> None:
        """Tests tiered cost calculation for Pro models."""
        # Pro model (<= 200k context)
        # Rates are 2.00 input, 0.20 cached, 12.00 output
        # input_tokens is total prompt size.
        # uncached = 10k - 5k = 5k.
        # cost = (5k * 2.0 + 5k * 0.2 + 2k * 12.0) / 1M = (10k + 1k + 24k) / 1M = 35k / 1M = 0.035
        cost_small = token_usage.calculate_cost(
            "gemini-3-pro-preview", 10_000, 5_000, 2_000
        )
        self.assertAlmostEqual(cost_small, 0.035)

        # Pro model (> 200k context)
        # Rates are 4.00 input, 0.40 cached, 18.00 output
        # uncached = 250k - 50k = 200k.
        # cost = (200k * 4.0 + 50k * 0.4 + 10k * 18.0) / 1M = (800k + 20k + 180k) / 1M = 1M / 1M = 1.0
        cost_large = token_usage.calculate_cost(
            "gemini-3-pro-preview", 250_000, 50_000, 10_000
        )
        self.assertAlmostEqual(cost_large, 1.0)

    def test_calculate_cost_fixed(self) -> None:
        """Tests fixed cost calculation for Flash models."""
        # Gemini 3 Flash: 0.50 input, 0.05 cached, 3.00 output
        # uncached = 1M - 500k = 500k.
        # cost = (500k * 0.5 + 500k * 0.05 + 100k * 3.0) / 1M = (250k + 25k + 300k) / 1M = 575k / 1M = 0.575
        cost_flash = token_usage.calculate_cost(
            "gemini-3-flash-preview", 1_000_000, 500_000, 100_000
        )
        self.assertAlmostEqual(cost_flash, 0.575)

    def test_get_date_range(self) -> None:
        """Tests named date range logic."""
        today = date(2026, 2, 5)  # A Thursday

        # Today
        start, end = token_usage.get_date_range("today", today_obj=today)
        self.assertEqual(start, "2026-02-05")
        self.assertEqual(end, "2026-02-05")

        # Yesterday
        start, end = token_usage.get_date_range("yesterday", today_obj=today)
        self.assertEqual(start, "2026-02-04")
        self.assertEqual(end, "2026-02-04")

        # This Week (starts Monday, Feb 2nd)
        start, end = token_usage.get_date_range("this-week", today_obj=today)
        self.assertEqual(start, "2026-02-02")
        self.assertEqual(end, "2026-02-05")

        # Last Week (Mon, Jan 26 - Sun, Feb 1st)
        start, end = token_usage.get_date_range("last-week", today_obj=today)
        self.assertEqual(start, "2026-01-26")
        self.assertEqual(end, "2026-02-01")

        # Since
        start, end = token_usage.get_date_range("since:2026-01-01", today_obj=today)
        self.assertEqual(start, "2026-01-01")
        self.assertEqual(end, "2026-02-05")

    def test_format_duration(self) -> None:
        """Tests human-readable duration formatting."""
        self.assertEqual(token_usage.format_duration(45), "45s")
        self.assertEqual(token_usage.format_duration(120), "2.0m")
        self.assertEqual(token_usage.format_duration(3600), "1.0h")
        self.assertEqual(token_usage.format_duration(5400), "1.5h")

        self.assertEqual(token_usage.format_duration_h(120), "2.0m")
        self.assertEqual(token_usage.format_duration_h(3600), "1.0h")

    def test_discover_session_files(self) -> None:
        """Tests recursive session file discovery."""
        with TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            # Standard structure
            (tmp_path / "uuid1" / "chats").mkdir(parents=True)
            f1 = tmp_path / "uuid1" / "chats" / "session-1.json"
            f1.touch()

            # Flat structure
            f2 = tmp_path / "uuid2" / "session-2.json"
            (tmp_path / "uuid2").mkdir(parents=True)
            f2.touch()

            # Non-session file
            (tmp_path / "other.txt").touch()

            files = token_usage.discover_session_files([tmp_path])
            self.assertEqual(len(files), 2)
            self.assertIn(f1, files)
            self.assertIn(f2, files)

    def test_filter_stats(self) -> None:
        """Tests date-based filtering of stats."""
        stats = {
            "2026-01-01": {"m1": token_usage.ModelStats()},
            "2026-01-15": {"m1": token_usage.ModelStats()},
            "2026-02-01": {"m1": token_usage.ModelStats()},
            "unknown": {"m1": token_usage.ModelStats()},
        }
        filtered = token_usage.filter_stats(stats, "2026-01-10", "2026-01-31")
        self.assertEqual(list(filtered.keys()), ["2026-01-15"])

    def test_since_logic(self) -> None:
        """Verifies the logic used for the --since flag."""
        from datetime import date, timedelta

        # Simulated 'since' date
        start_obj = date(2026, 1, 15)
        today_obj = date(2026, 1, 17)

        # Logic from main()
        date_filter = {
            (start_obj + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range((today_obj - start_obj).days + 1)
        }

        self.assertEqual(len(date_filter), 3)
        self.assertIn("2026-01-15", date_filter)
        self.assertIn("2026-01-16", date_filter)
        self.assertIn("2026-01-17", date_filter)

    def test_config_model_matching(self) -> None:
        """Tests that config correctly identifies models by substring."""
        config = token_usage.Config()
        pricing = token_usage.ModelPricing(token_usage.PricingTier(1.0, 0.1, 2.0))
        config.models["special-model"] = pricing

        # Exact match
        self.assertEqual(config.get_pricing("special-model"), pricing)
        # Substring/Versioned match
        self.assertEqual(config.get_pricing("special-model-001"), pricing)
        # Case insensitive
        self.assertEqual(config.get_pricing("SPECIAL-MODEL"), pricing)
        # Default
        self.assertEqual(config.get_pricing("unknown"), config.default_pricing)

    def test_print_report_raw(self) -> None:
        """Tests raw token count output mode."""
        stats = {
            "2026-01-20": {
                "m1": token_usage.ModelStats(
                    input_tokens=100, cached_tokens=50, output_tokens=30
                )
            }
        }
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        token_usage.print_report(stats, raw_tokens_only=True)
        sys.stdout = sys.__stdout__
        self.assertEqual(captured_output.getvalue().strip(), "180")

    def test_load_config_custom(self) -> None:
        """Verifies parsing of custom pricing.json."""
        with TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            pricing_file = tmp_path / ".gemini" / "pricing.json"
            pricing_file.parent.mkdir()

            custom_pricing = {
                "models": {
                    "simple-model": [1.0, 0.1, 5.0],
                    "tiered-model": {
                        "small_context": [2.0, 0.2, 10.0],
                        "large_context": [4.0, 0.4, 20.0],
                        "context_threshold": 100_000,
                    },
                }
            }
            with pricing_file.open("w") as f:
                json.dump(custom_pricing, f)

            with patch("pathlib.Path.home", return_value=tmp_path):
                config = token_usage.load_config()

                # Check simple model
                simple = config.get_pricing("simple-model")
                self.assertEqual(simple.small_context.input_rate, 1.0)
                self.assertIsNone(simple.large_context)

                # Check tiered model
                tiered = config.get_pricing("tiered-model")
                self.assertEqual(tiered.small_context.input_rate, 2.0)
                # Ensure large_context is present before checking rate
                self.assertIsNotNone(tiered.large_context)
                if tiered.large_context:
                    self.assertEqual(tiered.large_context.input_rate, 4.0)
                self.assertEqual(tiered.context_threshold, 100_000)

    def test_aggregate_usage_custom_dir(self) -> None:
        """Verifies that aggregate_usage handles a custom base_dir correctly."""
        with TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            session_file = tmp_path / "session-custom.json"

            session_data = {
                "sessionId": "custom-id",
                "startTime": "2026-03-01T12:00:00Z",
                "messages": [
                    {"type": "gemini", "model": "m1", "tokens": {"input": 100}}
                ],
            }
            with session_file.open("w") as f:
                json.dump(session_data, f)

            stats = token_usage.aggregate_usage(base_dir=tmp_path)
            self.assertIn("2026-03-01", stats)
            self.assertEqual(stats["2026-03-01"]["m1"].input_tokens, 100)

            # Check local cache
            self.assertTrue((tmp_path / "usage_cache.json").exists())

    def test_aggregation_keeps_deleted_files(self) -> None:
        """Verifies that stats are kept for files deleted from disk but present in cache."""
        with TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            chat_dir = tmp_path / "chats"
            chat_dir.mkdir(parents=True)

            session_data = {
                "sessionId": "test-deleted",
                "startTime": "2026-01-20T12:00:00Z",
                "messages": [
                    {
                        "type": "gemini",
                        "model": "gemini-3-flash",
                        "tokens": {"input": 100},
                    },
                ],
            }
            session_file = chat_dir / "session-del.json"
            with session_file.open("w") as f:
                json.dump(session_data, f)

            stats = token_usage.aggregate_usage(base_dir=tmp_path)
            self.assertEqual(stats["2026-01-20"]["gemini-3-flash"].input_tokens, 100)

            # Delete the file
            session_file.unlink()

            # Second run: file is gone, but cache has it
            stats_cached = token_usage.aggregate_usage(base_dir=tmp_path)
            self.assertEqual(
                stats_cached["2026-01-20"]["gemini-3-flash"].input_tokens, 100
            )

    def test_multi_day_session(self) -> None:
        """Verifies that token usage spanning multiple days is attributed correctly."""
        with TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            chat_dir = tmp_path / "chats"
            chat_dir.mkdir(parents=True)

            session_data = {
                "sessionId": "test-multi-day",
                "startTime": "2026-01-20T12:00:00Z",
                "messages": [
                    {
                        "type": "gemini",
                        "model": "gemini-3-flash",
                        "timestamp": "2026-01-20T12:00:10Z",
                        "tokens": {"input": 100},
                    },
                    {
                        "type": "gemini",
                        "model": "gemini-3-flash",
                        "timestamp": "2026-01-21T12:00:10Z",
                        "tokens": {"input": 50},
                    },
                ],
            }
            session_file = chat_dir / "session-1.json"
            with session_file.open("w") as f:
                json.dump(session_data, f)

            stats = token_usage.aggregate_usage(base_dir=tmp_path)
            self.assertIn("2026-01-20", stats)
            self.assertIn("2026-01-21", stats)
            self.assertEqual(stats["2026-01-20"]["gemini-3-flash"].input_tokens, 100)
            self.assertEqual(stats["2026-01-21"]["gemini-3-flash"].input_tokens, 50)

    def test_aggregation_size_invalidation(self) -> None:
        """Verifies that cache invalidates correctly when file size changes but mtime remains same."""
        with TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            chat_dir = tmp_path / "chats"
            chat_dir.mkdir(parents=True)

            session_file = chat_dir / "session-append.jsonl"

            # Initial write (need a fake session to establish the date first)
            init_line = json.dumps(
                {"sessionId": "test-append", "startTime": "2026-01-20T12:00:00Z"}
            )
            line1 = json.dumps(
                {
                    "type": "gemini",
                    "model": "m1",
                    "tokens": {"input": 10},
                }
            )
            with session_file.open("w") as f:
                f.write(init_line + "\n")
                f.write(line1 + "\n")

            # Fix mtime to a specific value
            fixed_mtime = time.time() - 1000
            os.utime(session_file, (fixed_mtime, fixed_mtime))

            stats = token_usage.aggregate_usage(base_dir=tmp_path)
            self.assertEqual(stats["2026-01-20"]["m1"].input_tokens, 10)

            # Append another line, but keep mtime EXACTLY the same
            line2 = json.dumps(
                {"type": "gemini", "model": "m1", "tokens": {"input": 20}}
            )
            with session_file.open("a") as f:
                f.write(line2 + "\n")

            os.utime(session_file, (fixed_mtime, fixed_mtime))

            # Re-aggregate, should pick up the new 20 tokens because size changed
            stats2 = token_usage.aggregate_usage(base_dir=tmp_path)
            self.assertEqual(stats2["2026-01-20"]["m1"].input_tokens, 30)

    def test_rolling_24h_filter(self) -> None:
        """Verifies that rolling 24-hour filter correctly excludes old messages."""
        with TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            chat_dir = tmp_path / "chats"
            chat_dir.mkdir(parents=True)

            # Current time: 2026-01-20T12:00:00Z
            now_ts = datetime(2026, 1, 20, 12, 0, 0, tzinfo=timezone.utc).timestamp()
            cutoff_ts = now_ts - 86400  # 24h ago

            session_data = {
                "sessionId": "test-24h",
                "startTime": "2026-01-19T10:00:00Z",
                "messages": [
                    {
                        "type": "gemini",
                        "model": "m1",
                        "timestamp": "2026-01-19T11:00:00Z", # >24h ago
                        "tokens": {"input": 100},
                    },
                    {
                        "type": "gemini",
                        "model": "m1",
                        "timestamp": "2026-01-19T13:00:00Z", # <24h ago
                        "tokens": {"input": 50},
                    },
                    {
                        "type": "gemini",
                        "model": "m1",
                        "timestamp": "2026-01-20T10:00:00Z", # <24h ago
                        "tokens": {"input": 30},
                    },
                ],
            }
            session_file = chat_dir / "session-24h.json"
            with session_file.open("w") as f:
                json.dump(session_data, f)

            # Test with since_timestamp
            stats = token_usage.aggregate_usage(base_dir=tmp_path, since_timestamp=cutoff_ts)

            # Should have 50 + 30 = 80 tokens.
            # 100 should be excluded.
            total_input = 0
            for day_stats in stats.values():
                for m_stats in day_stats.values():
                    total_input += m_stats.input_tokens

            self.assertEqual(total_input, 80)

    def test_fast_fail_lock(self) -> None:
        """Verifies that fast_fail flag causes immediate exit if lock cannot be acquired."""
        with TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            lock_file = tmp_path / "usage_cache.lock"

            # Pre-acquire the lock from another process context
            import fcntl

            lock_f = lock_file.open("a+")
            fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)

            try:
                # Should exit with code 2
                with self.assertRaises(SystemExit) as cm:
                    token_usage.aggregate_usage(base_dir=tmp_path, fast_fail=True)

                self.assertEqual(cm.exception.code, 2)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
                lock_f.close()


if __name__ == "__main__":
    unittest.main()
