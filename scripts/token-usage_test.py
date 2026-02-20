#!/usr/bin/env python3
"""Tests for token_usage.py logic."""

import io
import json
import os
import sys
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

# Add the scripts directory to path to import token_usage
sys.path.append(os.path.dirname(__file__))
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


if __name__ == "__main__":
    unittest.main()
