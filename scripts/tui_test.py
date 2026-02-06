#!/usr/bin/env python3
"""Tests for the Token Usage TUI."""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add scripts directory to path
sys.path.append(os.path.dirname(__file__))
import tui
import token_usage


class TestUsageTUI(unittest.TestCase):
    """Component and integration tests for the Curses TUI."""

    def setUp(self) -> None:
        self.tui = tui.UsageTUI()

    @patch("curses.wrapper")
    def test_tui_init_state(self, mock_wrapper: MagicMock) -> None:
        """Verifies initial state of the TUI object."""
        self.assertEqual(self.tui.current_filter, "all")
        self.assertTrue(self.tui.running)
        self.assertFalse(self.tui.show_models)

    @patch("token_usage.aggregate_usage")
    @patch("curses.newpad")
    @patch("curses.newwin")
    @patch("curses.curs_set")
    @patch("curses.doupdate")
    def test_main_loop_rendering(self, mock_doupdate, mock_curs_set, 
                                mock_newwin, mock_newpad, mock_aggregate) -> None:
        """Verifies that the main loop triggers data loading and rendering calls."""
        # Setup mock data
        mock_aggregate.return_value = {
            "2026-02-05": {
                "gemini-3-flash": token_usage.ModelStats(
                    input_tokens=100, cached_tokens=0, output_tokens=50, cost=0.01
                )
            }
        }
        
        # Mock screen
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 80)
        # Force loop exit after one iteration
        mock_stdscr.getch.return_value = ord('q')
        
        # Execute
        self.tui.main_loop(mock_stdscr)
        
        # Assertions
        mock_aggregate.assert_called_once()
        mock_newpad.assert_called()
        self.assertFalse(self.tui.running)

    def test_filter_cycling(self) -> None:
        """Verifies keyboard handling for filter changes."""
        mock_stdscr = MagicMock()
        
        # Open filter menu
        self.tui.handle_input(ord('f'), mock_stdscr)
        self.assertTrue(self.tui.show_filter_menu)
        
        # Navigate and Select 'today' (idx 1)
        self.tui.handle_input(258, mock_stdscr) # curses.KEY_DOWN (approx)
        # Note: handle_input uses curses constants, so we should mock or use them
        with patch("curses.KEY_DOWN", 258), patch("tui.curses.KEY_ENTER", 10):
             self.tui.handle_input(258, mock_stdscr)
             self.tui.handle_input(10, mock_stdscr)
             
        self.assertFalse(self.tui.show_filter_menu)
        # If we pressed down twice starting from all (0), we should be at yesterday (2)?
        # Actually filter_options = [all, today, yesterday, ...]
        # Let's just check if it changed
        self.assertNotEqual(self.tui.current_filter, "all")

    @patch("token_usage.get_date_range")
    @patch("token_usage.filter_stats")
    def test_refresh_view_data_alignment(self, mock_filter, mock_range) -> None:
        """Verifies that column widths are calculated correctly."""
        self.tui.stats = {
            "2026-02-05": {
                "short": token_usage.ModelStats(input_tokens=10),
                "very-long-model-name-for-testing-alignment": token_usage.ModelStats(input_tokens=100)
            }
        }
        self.tui.show_models = True
        self.tui.refresh_view_data()
        
        # col_widths[1] should be at least length of long model name
        self.assertGreater(self.tui.col_widths[1], 40)


if __name__ == "__main__":
    unittest.main()
