import json
import unittest
from pathlib import Path
import tempfile
import sys
import os
import io

# Add the scripts directory to path to import token-usage
sys.path.append(os.path.dirname(__file__))
import token_usage as tu


class TestTokenUsage(unittest.TestCase):
    def test_aggregation(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)
            chat_dir = tmp_path / "project1" / "chats"
            chat_dir.mkdir(parents=True)

            session_data = {
                "sessionId": "test-session",
                "startTime": "2026-01-20T12:00:00Z",
                "messages": [
                    {"type": "user", "content": "hello"},
                    {
                        "type": "gemini",
                        "model": "gemini-3-flash-preview",
                        "tokens": {
                            "input": 100,
                            "cached": 50,
                            "output": 20,
                            "thoughts": 10,
                        },
                    },
                ],
            }

            with (chat_dir / "session-1.json").open("w") as f:
                json.dump(session_data, f)

            stats = tu.aggregate_usage(base_dir=tmp_path)

            self.assertIn("2026-01-20", stats)
            flash_stats = stats["2026-01-20"]["gemini-3-flash-preview"]
            self.assertEqual(flash_stats["input"], 100)
            self.assertEqual(flash_stats["cached"], 50)
            self.assertEqual(flash_stats["output"], 30)  # 20 + 10
            self.assertEqual(len(flash_stats["sessions"]), 1)

    def test_cost_calculation(self):
        # Gemini 3 Flash: $0.50 Input, $0.05 Cache, $3.00 Output per 1M
        cost_flash = tu.calculate_cost(
            "gemini-3-flash-preview", 1_000_000, 1_000_000, 1_000_000
        )
        self.assertAlmostEqual(cost_flash, 0.50 + 0.05 + 3.00)

        # Gemini 3 Pro (<= 200k): $2.00 Input, $0.20 Cache, $12.00 Output per 1M
        cost_pro_small = tu.calculate_cost(
            "gemini-3-pro-preview", 100_000, 50_000, 10_000
        )
        expected_small = (100_000 * 2.00 + 50_000 * 0.20 + 10_000 * 12.00) / 1_000_000
        self.assertAlmostEqual(cost_pro_small, expected_small)

        # Gemini 3 Pro (> 200k): $4.00 Input, $0.40 Cache, $18.00 Output per 1M
        cost_pro_large = tu.calculate_cost("gemini-3-pro-preview", 200_001, 0, 0)
        expected_large = (200_001 * 4.00) / 1_000_000
        self.assertAlmostEqual(cost_pro_large, expected_large)

        # Gemini 2.5 Pro (<= 200k): $1.25 Input, $0.125 Cache, $10.00 Output per 1M
        cost_25_pro_small = tu.calculate_cost(
            "gemini-2.5-pro", 100_000, 0, 0
        )
        self.assertAlmostEqual(cost_25_pro_small, 0.125)

        # Gemini 2.5 Pro (> 200k): $2.50 Input, $0.25 Cache, $15.00 Output per 1M
        cost_25_pro_large = tu.calculate_cost(
            "gemini-2.5-pro", 200_001, 0, 0
        )
        self.assertAlmostEqual(cost_25_pro_large, (200_001 * 2.50) / 1_000_000)

        # Gemini 2.5 Flash: $0.30 Input, $0.03 Cache, $2.50 Output per 1M
        cost_25_flash = tu.calculate_cost(
            "gemini-2.5-flash", 1_000_000, 1_000_000, 1_000_000
        )
        self.assertAlmostEqual(cost_25_flash, 0.30 + 0.03 + 2.50)

        # Gemini 2.5 Flash Lite: $0.10 Input, $0.01 Cache, $0.40 Output per 1M
        cost_25_flash_lite = tu.calculate_cost(
            "gemini-2.5-flash-lite", 1_000_000, 1_000_000, 1_000_000
        )
        self.assertAlmostEqual(cost_25_flash_lite, 0.10 + 0.01 + 0.40)

        # Gemini 2.0 Flash: $0.10 Input, $0.025 Cache, $0.40 Output per 1M
        cost_20_flash = tu.calculate_cost(
            "gemini-2.0-flash", 1_000_000, 1_000_000, 1_000_000
        )
        self.assertAlmostEqual(cost_20_flash, 0.10 + 0.025 + 0.40)

        # Gemini 2.0 Flash Lite: $0.075 Input, $0.0 Cache, $0.30 Output per 1M
        cost_20_flash_lite = tu.calculate_cost(
            "gemini-2.0-flash-lite", 1_000_000, 1_000_000, 1_000_000
        )
        self.assertAlmostEqual(cost_20_flash_lite, 0.075 + 0.30)

        # General Flash fallback: $0.50 Input, $0.05 Cache, $3.00 Output per 1M
        cost_general_flash = tu.calculate_cost(
            "some-other-flash", 1_000_000, 1_000_000, 1_000_000
        )
        self.assertAlmostEqual(cost_general_flash, 0.50 + 0.05 + 3.00)

        # Unknown model (defaults to Pro pricing): $2.00 Input, $0.20 Cache, $12.00 Output per 1M
        cost_unknown = tu.calculate_cost(
            "unknown-model", 100_000, 50_000, 10_000
        )
        expected_unknown = (100_000 * 2.00 + 50_000 * 0.20 + 10_000 * 12.00) / 1_000_000
        self.assertAlmostEqual(cost_unknown, expected_unknown)


class TestDateFiltering(unittest.TestCase):
    def test_get_date_range_yesterday(self):
        # Mocking datetime.now is tricky, let's see how we can test this.
        # Maybe we can pass an optional 'today' to get_date_range for testing.
        from datetime import date
        today = date(2026, 2, 5)  # Thursday
        
        # yesterday
        start, end = tu.get_date_range("yesterday", today=today)
        self.assertEqual(start, "2026-02-04")
        self.assertEqual(end, "2026-02-04")

    def test_get_date_range_this_week(self):
        from datetime import date
        today = date(2026, 2, 5)  # Thursday
        
        # this-week: Monday to today
        start, end = tu.get_date_range("this-week", today=today)
        self.assertEqual(start, "2026-02-02")
        self.assertEqual(end, "2026-02-05")

    def test_get_date_range_last_week(self):
        from datetime import date
        today = date(2026, 2, 5)  # Thursday
        
        # last-week: Previous Monday to previous Sunday
        start, end = tu.get_date_range("last-week", today=today)
        self.assertEqual(start, "2026-01-26")
        self.assertEqual(end, "2026-02-01")

    def test_get_date_range_this_month(self):
        from datetime import date
        today = date(2026, 2, 5)
        
        # this-month: 1st of month to today
        start, end = tu.get_date_range("this-month", today=today)
        self.assertEqual(start, "2026-02-01")
        self.assertEqual(end, "2026-02-05")

    def test_get_date_range_last_month(self):
        from datetime import date
        today = date(2026, 2, 5)
        
        # last-month: 1st to last of previous month
        start, end = tu.get_date_range("last-month", today=today)
        self.assertEqual(start, "2026-01-01")
        self.assertEqual(end, "2026-01-31")

    def test_get_date_range_custom(self):
        # custom range: "YYYY-MM-DD:YYYY-MM-DD"
        start, end = tu.get_date_range("2026-01-01:2026-01-15")
        self.assertEqual(start, "2026-01-01")
        self.assertEqual(end, "2026-01-15")

    def test_filter_stats(self):
        stats = {
            "2026-01-01": {"model1": {"input": 10}},
            "2026-01-05": {"model1": {"input": 20}},
            "2026-01-10": {"model1": {"input": 30}},
            "unknown": {"model1": {"input": 40}}
        }
        filtered = tu.filter_stats(stats, "2026-01-05", "2026-01-10")
        self.assertEqual(len(filtered), 2)
        self.assertIn("2026-01-05", filtered)
        self.assertIn("2026-01-10", filtered)
        self.assertNotIn("2026-01-01", filtered)
        self.assertNotIn("unknown", filtered)


class TestReporting(unittest.TestCase):
    def setUp(self):
        self.stats = {
            "2026-02-01": {
                "gemini-3-flash-long-model-name-testing": {
                    "sessions": {"s1"},
                    "input": 1000000,
                    "cached": 0,
                    "output": 500000,
                    "cost": 2.0,
                },
                "gemini-3-pro": {
                    "sessions": {"s2"},
                    "input": 1000000,
                    "cached": 0,
                    "output": 500000,
                    "cost": 8.0,
                }
            }
        }

    def test_print_report_with_models(self):
        captured_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output
        try:
            tu.print_report(self.stats, show_models=True)
        finally:
            sys.stdout = old_stdout
        output = captured_output.getvalue()
        
        self.assertIn("gemini-3-flash-long-model-name-testing", output)
        self.assertIn("gemini-3-pro", output)
        # Check for non-truncated name in TOTALS section
        self.assertIn("TOTALS (gemini-3-flash-long-model-name-testing)", output)
        # Total tokens: 1.5M + 1.5M = 3.0M
        self.assertIn("3,000,000", output)
        # Total cost: 2.0 + 8.0 = 10.0
        self.assertIn("$   10.00", output)

    def test_print_summary_statistics_with_models(self):
        captured_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output
        try:
            tu.print_summary_statistics(self.stats, show_models=True)
        finally:
            sys.stdout = old_stdout
        output = captured_output.getvalue()
        
        self.assertIn("SUMMARY BY MODEL", output)
        # The name might be truncated if it is too long, but we should at least see a good part of it.
        # After my fix to increase width to 40, it should be fully there.
        self.assertIn("gemini-3-flash-long-model-name-testing", output)
        self.assertIn("TOTAL TOKENS", output)
        self.assertIn("TOTAL COST", output)
        self.assertIn("1,500,000", output)
        self.assertIn("$      2.00", output)
        self.assertIn("$      8.00", output)

    def test_print_summary_statistics_tabular(self):
        captured_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output
        try:
            tu.print_summary_statistics(self.stats, show_models=True)
        finally:
            sys.stdout = old_stdout
        output = captured_output.getvalue()
        
        # Check general summary table headers
        self.assertIn("PERIOD", output)
        self.assertIn("TOKENS", output)
        self.assertIn("COST", output)
        
        # Check model summary table headers
        self.assertIn("MODEL", output)
        self.assertIn("DAYS", output)
        self.assertIn("TOTAL TOKENS", output)
        self.assertIn("AVG TOKENS/D", output)
        self.assertIn("TOTAL COST", output)
        self.assertIn("AVG COST/D", output)


if __name__ == "__main__":
    unittest.main()
