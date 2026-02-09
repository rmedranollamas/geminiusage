#!/usr/bin/env python3
"""Tests for token_usage.py logic."""

import io
import json
import os
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

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
            self.assertEqual(model_stats.input_tokens, 100)
            self.assertEqual(model_stats.output_tokens, 30)  # 20 + 10
            self.assertEqual(model_stats.cached_tokens, 50)

            # Verify cache file was created
            cache_file = tmp_path / "usage_cache.json"
            self.assertTrue(cache_file.exists())

            # Second run: should use cache
            stats_cached = token_usage.aggregate_usage(base_dir=tmp_path)
            self.assertEqual(stats_cached["2026-01-20"]["gemini-3-flash"].input_tokens, 100)
            self.assertEqual(stats_cached["2026-01-20"]["gemini-3-flash"].cached_tokens, 50)

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
                    {
                        "type": "gemini",
                        "model": "gemini-3-flash",
                        "tokens": None
                    },
                    None # Malformed message
                ]
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
        cost_small = token_usage.calculate_cost("gemini-3-pro", 100_000, 0, 0)
        self.assertAlmostEqual(cost_small, 0.20)  # (100k * 2.00) / 1M

        # Pro model (> 200k context)
        cost_large = token_usage.calculate_cost("gemini-3-pro", 200_001, 0, 0)
        self.assertAlmostEqual(cost_large, (200_001 * 4.00) / 1_000_000)

    def test_calculate_cost_models(self) -> None:
        """Tests specific cost rates for different models."""
        # Test input cost per 1M tokens within small context
        models = [
            ("gemini-3-flash", 100_000, 0.50),
            ("gemini-2.5-pro", 100_000, 1.25),
            ("gemini-2.5-flash-lite", 100_000, 0.10),
            ("gemini-2.0-flash", 100_000, 0.10),
        ]
        for model_name, input_tokens, expected_rate in models:
            with self.subTest(model=model_name):
                # We pass 1,000,000 tokens to get the rate directly in USD
                cost = token_usage.calculate_cost(model_name, 1_000_000, 0, 0)
                # But wait, if we pass 1M, it exceeds the 200k threshold.
                # Let's pass a small amount and multiply.
                cost_small = token_usage.calculate_cost(model_name, input_tokens, 0, 0)
                calculated_rate = cost_small * (1_000_000 / input_tokens)
                self.assertAlmostEqual(calculated_rate, expected_rate, places=4)


class TestDateFiltering(unittest.TestCase):
    """Tests for date range generation and filtering."""

    def test_get_date_range_named(self) -> None:
        """Tests standard named ranges like 'yesterday' or 'this-week'."""
        today_obj = date(2026, 2, 5)  # Thursday

        test_cases = {
            "yesterday": ("2026-02-04", "2026-02-04"),
            "this-week": ("2026-02-02", "2026-02-05"),
            "last-week": ("2026-01-26", "2026-02-01"),
            "this-month": ("2026-02-01", "2026-02-05"),
            "last-month": ("2026-01-01", "2026-01-31"),
        }

        for name, (expected_start, expected_end) in test_cases.items():
            with self.subTest(range=name):
                start, end = token_usage.get_date_range(name, today_obj=today_obj)
                self.assertEqual(start, expected_start)
                self.assertEqual(end, expected_end)

    def test_filter_stats_logic(self) -> None:
        """Tests the actual filtering of aggregated stats dict."""
        stats = {
            "2026-01-01": {"m": token_usage.ModelStats(input_tokens=10)},
            "2026-01-10": {"m": token_usage.ModelStats(input_tokens=20)},
        }
        filtered = token_usage.filter_stats(stats, "2026-01-05", "2026-01-15")
        self.assertEqual(len(filtered), 1)
        self.assertIn("2026-01-10", filtered)


class TestReporting(unittest.TestCase):
    """Tests for CLI reporting output."""

    def setUp(self) -> None:
        self.stats = {
            "2026-02-01": {
                "gemini-3-pro": token_usage.ModelStats(
                    sessions={"s1"}, input_tokens=1000, cost=0.01
                )
            }
        }

    def test_print_report_tabular(self) -> None:
        """Verifies tabular report format."""
        output = io.StringIO()
        with patch("sys.stdout", output):
            token_usage.print_report(self.stats)
        
        text = output.getvalue()
        self.assertIn("DATE", text)
        self.assertIn("2026-02-01", text)
        self.assertIn("1,000", text)

    def test_summary_statistics_calculations(self) -> None:
        """Verifies aggregate summary math."""
        output = io.StringIO()
        with patch("sys.stdout", output):
            token_usage.print_summary_statistics(self.stats)
        
        text = output.getvalue()
        self.assertIn("SUMMARY STATISTICS", text)
        self.assertIn("All Time", text)

    def test_main_cli_dispatch(self) -> None:
        """Verifies the main function executes with mocked arguments."""
        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = MagicMock(
                model=False, raw=True, today=True, yesterday=False,
                this_week=False, last_week=False, this_month=False,
                last_month=False, date_range=None
            )
            with patch("token_usage.aggregate_usage") as mock_agg:
                mock_agg.return_value = {}
                output = io.StringIO()
                with patch("sys.stdout", output):
                    token_usage.main()
                self.assertEqual(output.getvalue().strip(), "0")


if __name__ == "__main__":
    unittest.main()
