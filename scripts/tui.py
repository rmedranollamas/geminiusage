#!/usr/bin/env python3
"""Interactive Terminal User Interface for Gemini Token Usage."""

import argparse
import curses
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import token_usage


class UsageTUI:
    """Main application class for the Token Usage TUI."""

    # Layout Constants
    HEADER_H = 1
    FOOTER_H = 1
    COL_HEADER_Y = 1
    MENU_WIDTH = 24
    MIN_TOTALS_H = 3

    def __init__(self, base_dir: Optional[str] = None, initial_filter: str = "all"):
        """Initializes the TUI state."""
        self.base_dir = Path(base_dir) if base_dir else None
        self.stats: Dict[str, Dict[str, Any]] = {}
        self.current_filter = initial_filter
        self.show_models = False
        self.running = True
        self.scroll_y = 0
        self.selected_row = 0
        self.view_rows: List[List[str]] = []
        self.view_data: List[Tuple[str, Union[str, Tuple[str, str]]]] = []
        self.col_widths: List[int] = []
        self.totals = token_usage.ModelStats()
        self.model_totals: Dict[str, token_usage.ModelStats] = {}
        self.filter_options = [
            "all",
            "today",
            "yesterday",
            "this-week",
            "last-week",
            "this-month",
            "last-month",
        ]
        if self.current_filter not in self.filter_options:
            self.filter_options.append(self.current_filter)

        self.show_filter_menu = False
        self.menu_selected = self.filter_options.index(self.current_filter)
        self.table_pad: Optional[Any] = None
        self.totals_win: Optional[Any] = None
        self.data_dirty = True
        self.ui_dirty = True

        # Auto-refresh settings
        self.last_refresh = time.time()
        self.refresh_interval = 30

    def load_data(self) -> None:
        """Loads usage data and refreshes the view."""
        self.stats = token_usage.aggregate_usage(self.base_dir)
        self.last_refresh = time.time()
        self.refresh_view_data()
        self.data_dirty = True
        self.ui_dirty = True

    def refresh_view_data(self) -> None:
        """Processes raw stats into displayable rows and calculates column widths."""
        self.view_rows = []
        self.view_data = []
        self.totals = token_usage.ModelStats()
        self.model_totals = {}

        filtered_stats = self.stats
        if self.current_filter != "all":
            start, end = token_usage.get_date_range(self.current_filter)
            if start and end:
                filtered_stats = token_usage.filter_stats(self.stats, start, end)

        # 1. Aggregate totals and model-specific totals
        for day in sorted(filtered_stats.keys(), reverse=True):
            if day == "unknown":
                continue
            for model, s in filtered_stats[day].items():
                if model not in self.model_totals:
                    self.model_totals[model] = token_usage.ModelStats()

                self.model_totals[model].add(s)
                self.totals.add(s)

        # 2. Build data rows for the main table
        for day in sorted(filtered_stats.keys(), reverse=True):
            if day == "unknown":
                continue
            if not self.show_models:
                day_models = filtered_stats[day].values()
                day_stats = token_usage.ModelStats()
                for s in day_models:
                    day_stats.add(s)

                self.view_rows.append(
                    [
                        day,
                        str(len(day_stats.sessions)),
                        token_usage.format_duration(day_stats.duration_seconds),
                        f"{day_stats.input_tokens:,}",
                        f"{day_stats.cached_tokens:,}",
                        f"{day_stats.output_tokens:,}",
                        f"{day_stats.total_tokens:,}",
                        f"${day_stats.cost:,.2f}",
                    ]
                )
            else:
                for model in sorted(filtered_stats[day].keys()):
                    s = filtered_stats[day][model]
                    self.view_rows.append(
                        [
                            day,
                            model,
                            str(len(s.sessions)),
                            token_usage.format_duration(s.duration_seconds),
                            f"{s.input_tokens:,}",
                            f"{s.cached_tokens:,}",
                            f"{s.output_tokens:,}",
                            f"{s.total_tokens:,}",
                            f"${s.cost:,.2f}",
                        ]
                    )

        # 3. Calculate dynamic column widths
        header = (
            [
                "DATE",
                "MODEL",
                "SESS",
                "FOCUS",
                "INPUT",
                "CACHED",
                "OUTPUT",
                "TOTAL",
                "COST",
            ]
            if self.show_models
            else ["DATE", "SESS", "FOCUS", "INPUT", "CACHED", "OUTPUT", "TOTAL", "COST"]
        )

        # Start with header widths
        self.col_widths = [len(h) for h in header]

        # Update with data row widths
        for row in self.view_rows:
            for i, val in enumerate(row):
                self.col_widths[i] = max(self.col_widths[i], len(val))

        # Update with potential totals widths
        if self.show_models:
            for model in self.model_totals:
                self.col_widths[1] = max(self.col_widths[1], len(f"TOTAL ({model})"))

        # 4. Generate formatted lines
        for row in self.view_rows:
            line = ""
            for i, val in enumerate(row):
                align = "<" if i < (2 if self.show_models else 1) else ">"
                line += f"{val:{align}{self.col_widths[i]}}  "
            self.view_data.append((line.rstrip(), row[0]))

    def draw_header(self, stdscr: Any) -> None:
        """Draws the top status bar."""
        _, w = stdscr.getmaxyx()
        model_status = "ON" if self.show_models else "OFF"

        # Calculate countdown
        time_since_refresh = time.time() - self.last_refresh
        countdown = max(0, int(self.refresh_interval - time_since_refresh))

        header = (
            f" Gemini Token Usage TUI | Filter: [{self.current_filter}] | "
            f"Models: {model_status} | Refresh in {countdown}s | "
            f"{datetime.now().strftime('%H:%M:%S')} "
        )
        stdscr.attron(curses.A_REVERSE)
        try:
            stdscr.addstr(0, 0, header.ljust(w)[: w - 1])
        except curses.error:
            pass
        stdscr.attroff(curses.A_REVERSE)

    def draw_totals(self, stdscr: Any, start_row: int, height: int) -> None:
        """Draws the bordered totals panel."""
        h, w = stdscr.getmaxyx()
        if height < self.MIN_TOTALS_H:
            height = self.MIN_TOTALS_H

        if not self.totals_win:
            self.totals_win = curses.newwin(height, w, start_row, 0)
        else:
            self.totals_win.mvwin(start_row, 0)
            self.totals_win.resize(height, w)

        self.totals_win.erase()
        self.totals_win.box()
        self.totals_win.attron(curses.A_BOLD)
        self.totals_win.addstr(0, 2, f" TOTALS ({self.current_filter}) ")
        self.totals_win.attroff(curses.A_BOLD)

        label_col_width = (
            self.col_widths[0] + 2 + self.col_widths[1]
            if self.show_models
            else self.col_widths[0]
        )

        # Robust column indexing based on total number of columns
        def format_total_line(label: str, stats: token_usage.ModelStats) -> str:
            num_cols = len(self.col_widths)
            # Cost, Total, Output, Cached, Input, Focus are the last 6 columns
            cost_idx = num_cols - 1
            total_idx = num_cols - 2
            out_idx = num_cols - 3
            cached_idx = num_cols - 4
            input_idx = num_cols - 5
            focus_idx = num_cols - 6

            parts = [f"{label:<{label_col_width}}"]
            # Skip the 'SESS' column which is before FOCUS
            sess_idx = 2 if self.show_models else 1
            parts.append(f"{'':>{self.col_widths[sess_idx]}}")

            parts.append(
                f"{token_usage.format_duration(stats.duration_seconds):>{self.col_widths[focus_idx]}}"
            )
            parts.append(f"{stats.input_tokens:>{self.col_widths[input_idx]},}")
            parts.append(f"{stats.cached_tokens:>{self.col_widths[cached_idx]},}")
            parts.append(f"{stats.output_tokens:>{self.col_widths[out_idx]},}")
            parts.append(f"{stats.total_tokens:>{self.col_widths[total_idx]},}")

            cost_str = f"${stats.cost:,.2f}"
            parts.append(f"{cost_str:>{self.col_widths[cost_idx]}}")
            return "  ".join(parts)

        row_idx = 1
        if self.show_models:
            sorted_models = sorted(self.model_totals.keys())
            for model in sorted_models:
                if row_idx >= height - 2:
                    break
                line = format_total_line(
                    f"TOTAL ({model[:30]})", self.model_totals[model]
                )
                self.totals_win.addstr(row_idx, 1, line[: w - 2])
                row_idx += 1
            if row_idx < height - 1:
                self.totals_win.addstr(row_idx, 1, ("-" * (w - 2))[: w - 2])
                row_idx += 1

        if row_idx < height - 1:
            line = format_total_line("GRAND TOTAL (ALL)", self.totals)
            self.totals_win.attron(curses.A_BOLD)
            self.totals_win.addstr(row_idx, 1, line[: w - 2])
            self.totals_win.attroff(curses.A_BOLD)

        self.totals_win.refresh()

    def draw_filter_menu(self, stdscr: Any) -> None:
        """Draws a centered popup menu for filter selection."""
        h, w = stdscr.getmaxyx()
        menu_h = len(self.filter_options) + 2
        menu_w = self.MENU_WIDTH
        start_y = (h - menu_h) // 2
        start_x = (w - menu_w) // 2

        win = curses.newwin(menu_h, menu_w, start_y, start_x)
        win.box()
        win.addstr(0, 2, " Select Filter ")

        for i, option in enumerate(self.filter_options):
            if i == self.menu_selected:
                win.attron(curses.A_REVERSE)
            win.addstr(i + 1, 2, option.center(menu_w - 4))
            if i == self.menu_selected:
                win.attroff(curses.A_REVERSE)

        win.refresh()

    def draw_footer(self, stdscr: Any) -> None:
        """Draws the bottom command legend."""
        h, w = stdscr.getmaxyx()
        footer = " [Q] Quit | [R] Refresh | [M] Models | [F] Filter | [P] Pricing | [UP/DOWN] Select "
        stdscr.attron(curses.A_REVERSE)
        try:
            stdscr.addstr(h - 1, 0, footer.ljust(w)[: w - 1])
        except curses.error:
            pass
        stdscr.attroff(curses.A_REVERSE)

    def edit_pricing(self, stdscr: Any) -> None:
        """Launches the system editor to modify pricing config."""
        pricing_path = Path.home() / ".gemini" / "pricing.json"
        pricing_path.parent.mkdir(parents=True, exist_ok=True)
        if not pricing_path.exists():
            with pricing_path.open("w", encoding="utf-8") as f:
                json.dump({"models": {}}, f, indent=2)

        curses.def_shell_mode()
        stdscr.clear()
        stdscr.refresh()
        editor = os.environ.get("EDITOR", "vi")
        subprocess.call([editor, str(pricing_path)])
        curses.reset_shell_mode()

        token_usage.reload_config()
        self.load_data()
        self.table_pad = None

    def handle_input(self, key: int, stdscr: Any) -> None:
        """Handles user keyboard input."""
        if self.show_filter_menu:
            if key == curses.KEY_UP:
                self.menu_selected = (self.menu_selected - 1) % len(self.filter_options)
                self.ui_dirty = True
            elif key == curses.KEY_DOWN:
                self.menu_selected = (self.menu_selected + 1) % len(self.filter_options)
                self.ui_dirty = True
            elif key in [10, 13, curses.KEY_ENTER]:
                self.current_filter = self.filter_options[self.menu_selected]
                self.show_filter_menu = False
                self.selected_row = 0
                self.scroll_y = 0
                self.refresh_view_data()
                self.table_pad = None
                self.data_dirty = True
                self.ui_dirty = True
            elif key in [27, ord("f"), ord("F")]:
                self.show_filter_menu = False
                self.ui_dirty = True
            return

        if key in [ord("q"), ord("Q")]:
            self.running = False
        elif key in [ord("r"), ord("R")]:
            self.load_data()
            self.table_pad = None
        elif key in [ord("p"), ord("P")]:
            self.edit_pricing(stdscr)
        elif key in [ord("f"), ord("F")]:
            self.show_filter_menu = True
            self.menu_selected = self.filter_options.index(self.current_filter)
            self.ui_dirty = True
        elif key in [ord("m"), ord("M")]:
            self.show_models = not self.show_models
            self.selected_row = 0
            self.scroll_y = 0
            self.refresh_view_data()
            self.table_pad = None
            self.data_dirty = True
            self.ui_dirty = True
        elif key == curses.KEY_UP:
            if self.selected_row > 0:
                self.selected_row -= 1
                self.ui_dirty = True
        elif key == curses.KEY_DOWN:
            if self.selected_row < len(self.view_data) - 1:
                self.selected_row += 1
                self.ui_dirty = True
        elif key == curses.KEY_PPAGE:
            new_row = max(0, self.selected_row - 10)
            if new_row != self.selected_row:
                self.selected_row = new_row
                self.ui_dirty = True
        elif key == curses.KEY_NPAGE:
            new_row = min(len(self.view_data) - 1, self.selected_row + 10)
            if new_row != self.selected_row:
                self.selected_row = new_row
                self.ui_dirty = True
        elif key == curses.KEY_RESIZE:
            self.table_pad = None
            self.totals_win = None
            self.ui_dirty = True

    def main_loop(self, stdscr: Any) -> None:
        """Core application loop."""
        curses.curs_set(0)
        stdscr.keypad(True)
        # Set a timeout for getch() to allow for auto-refresh
        stdscr.timeout(1000)

        self.load_data()
        self.table_pad = None

        while self.running:
            # Check for auto-refresh
            if time.time() - self.last_refresh >= self.refresh_interval:
                self.load_data()
                self.table_pad = None

            h, w = stdscr.getmaxyx()
            if h < 10 or w < 40:
                stdscr.erase()
                stdscr.addstr(0, 0, "Terminal too small")
                stdscr.refresh()
                key = stdscr.getch()
                if key in [ord("q"), ord("Q")]:
                    self.running = False
                continue

            # 1. Calculate layout
            totals_h = self.MIN_TOTALS_H
            if self.show_models:
                totals_h = min(len(self.model_totals) + 4, h // 3)

            table_y_start = 2
            table_y_end = h - totals_h - 2
            table_h = table_y_end - table_y_start + 1

            # Sync scrolling
            if self.selected_row < self.scroll_y:
                self.scroll_y = self.selected_row
                self.ui_dirty = True
            elif self.selected_row >= self.scroll_y + table_h:
                self.scroll_y = self.selected_row - table_h + 1
                self.ui_dirty = True

            if self.ui_dirty:
                stdscr.erase()
                self.draw_header(stdscr)
                self.draw_footer(stdscr)

                # 2. Draw static table header
                header_cols = (
                    [
                        "DATE",
                        "MODEL",
                        "SESS",
                        "INPUT",
                        "CACHED",
                        "OUTPUT",
                        "TOTAL",
                        "COST",
                    ]
                    if self.show_models
                    else ["DATE", "SESS", "INPUT", "CACHED", "OUTPUT", "TOTAL", "COST"]
                )
                col_header = ""
                for i, col in enumerate(header_cols):
                    align = "<" if i < (2 if self.show_models else 1) else ">"
                    col_header += f"{col:{align}{self.col_widths[i]}}  "
                try:
                    stdscr.addstr(self.COL_HEADER_Y, 0, col_header[: w - 1])
                except curses.error:
                    pass

                # 3. Handle pad creation and data rendering
                if not self.table_pad:
                    self.table_pad = curses.newpad(
                        max(len(self.view_data) + 1, 100), 256
                    )
                    self.data_dirty = True

                if self.data_dirty:
                    self.table_pad.erase()
                    for i, (line, _) in enumerate(self.view_data):
                        if i == self.selected_row:
                            self.table_pad.attron(curses.A_REVERSE)
                        self.table_pad.addstr(i, 0, line)
                        if i == self.selected_row:
                            self.table_pad.attroff(curses.A_REVERSE)
                    self.data_dirty = False
                else:
                    # Just update the highlighting if selection changed but data didn't
                    # (This is a simplified optimization, redrawing the whole pad is still better than every loop)
                    self.table_pad.erase()
                    for i, (line, _) in enumerate(self.view_data):
                        if i == self.selected_row:
                            self.table_pad.attron(curses.A_REVERSE)
                        self.table_pad.addstr(i, 0, line)
                        if i == self.selected_row:
                            self.table_pad.attroff(curses.A_REVERSE)

                # 5. Refresh screen
                stdscr.noutrefresh()
                if table_h > 0:
                    self.table_pad.noutrefresh(
                        self.scroll_y, 0, table_y_start, 0, table_y_end, w - 1
                    )

                self.draw_totals(stdscr, h - totals_h - 1, totals_h)

                if self.show_filter_menu:
                    self.draw_filter_menu(stdscr)

                curses.doupdate()
                self.ui_dirty = False
            else:
                # Still draw header to update countdown if nothing else changed
                self.draw_header(stdscr)
                stdscr.refresh()

            # 6. Process input (non-blocking thanks to timeout)
            try:
                key = stdscr.getch()
            except curses.error:
                key = -1

            if key != -1:
                self.handle_input(key, stdscr)
            else:
                # If no key, and stdin is no longer a tty, we should probably exit
                # to avoid an infinite loop of 100% CPU if the terminal is gone.
                if not os.isatty(0):
                    self.running = False


def main() -> None:
    """TUI Entry point."""
    parser = argparse.ArgumentParser(description="Interactive Gemini Token Usage TUI")
    parser.add_argument(
        "dir",
        nargs="?",
        default=None,
        help="Optional path to search for session files.",
    )
    parser.add_argument(
        "--since", help="Usage since a specific date (YYYY-MM-DD), inclusive."
    )
    args = parser.parse_args()

    initial_filter = "all"
    if args.since:
        initial_filter = f"since:{args.since}"

    tui = UsageTUI(base_dir=args.dir, initial_filter=initial_filter)
    curses.wrapper(tui.main_loop)


if __name__ == "__main__":
    main()
