#!/usr/bin/env python3
"""Interactive Terminal User Interface for Gemini Token Usage."""

import curses
import json
import os
import subprocess
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

    def __init__(self):
        """Initializes the TUI state."""
        self.stats: Dict[str, Dict[str, Any]] = {}
        self.current_filter = "all"
        self.show_models = False
        self.running = True
        self.scroll_y = 0
        self.selected_row = 0
        self.view_rows: List[List[str]] = []
        self.view_data: List[Tuple[str, Union[str, Tuple[str, str]]]] = []
        self.col_widths: List[int] = []
        self.totals: Dict[str, Union[int, float]] = {
            "input": 0, "cached": 0, "output": 0, "cost": 0.0
        }
        self.model_totals: Dict[str, Dict[str, Union[int, float]]] = {}
        self.filter_options = [
            "all", "today", "yesterday", "this-week", "last-week", 
            "this-month", "last-month"
        ]
        self.show_filter_menu = False
        self.menu_selected = 0
        self.table_pad: Optional[Any] = None

    def load_data(self) -> None:
        """Loads usage data and refreshes the view."""
        self.stats = token_usage.aggregate_usage()
        self.refresh_view_data()

    def refresh_view_data(self) -> None:
        """Processes raw stats into displayable rows and calculates column widths."""
        self.view_rows = []
        self.view_data = []
        self.totals = {"input": 0, "cached": 0, "output": 0, "cost": 0.0}
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
                    self.model_totals[model] = {
                        "input": 0, "cached": 0, "output": 0, "cost": 0.0
                    }
                
                # ModelStats from token_usage uses .input_tokens etc.
                self.model_totals[model]["input"] += s.input_tokens
                self.model_totals[model]["cached"] += s.cached_tokens
                self.model_totals[model]["output"] += s.output_tokens
                self.model_totals[model]["cost"] += s.cost

                self.totals["input"] += s.input_tokens
                self.totals["cached"] += s.cached_tokens
                self.totals["output"] += s.output_tokens
                self.totals["cost"] += s.cost

        # 2. Build data rows for the main table
        for day in sorted(filtered_stats.keys(), reverse=True):
            if day == "unknown":
                continue
            if not self.show_models:
                day_models = filtered_stats[day].values()
                inp = sum(s.input_tokens for s in day_models)
                cache = sum(s.cached_tokens for s in day_models)
                out = sum(s.output_tokens for s in day_models)
                cost = sum(s.cost for s in day_models)
                sess: Set[str] = set()
                for s in day_models:
                    sess.update(s.sessions)
                
                total = inp + cache + out
                self.view_rows.append([
                    day, str(len(sess)), f"{inp:,}", f"{cache:,}", 
                    f"{out:,}", f"{total:,}", f"${cost:,.2f}"
                ])
            else:
                for model in sorted(filtered_stats[day].keys()):
                    s = filtered_stats[day][model]
                    total = s.input_tokens + s.cached_tokens + s.output_tokens
                    self.view_rows.append([
                        day, model, str(len(s.sessions)), f"{s.input_tokens:,}", 
                        f"{s.cached_tokens:,}", f"{s.output_tokens:,}", f"{total:,}", 
                        f"${s.cost:,.2f}"
                    ])

        # 3. Calculate dynamic column widths
        header = (["DATE", "MODEL", "SESS", "INPUT", "CACHED", "OUTPUT", "TOTAL", "COST"] 
                  if self.show_models else ["DATE", "SESS", "INPUT", "CACHED", "OUTPUT", "TOTAL", "COST"])
        
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
        header = (f" Gemini Token Usage TUI | Filter: [{self.current_filter}] | "
                  f"Models: {model_status} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")
        stdscr.attron(curses.A_REVERSE)
        try:
            stdscr.addstr(0, 0, header.ljust(w)[:w-1])
        except curses.error:
            pass
        stdscr.attroff(curses.A_REVERSE)

    def draw_totals(self, stdscr: Any, start_row: int, height: int) -> None:
        """Draws the bordered totals panel."""
        h, w = stdscr.getmaxyx()
        if height < self.MIN_TOTALS_H:
            height = self.MIN_TOTALS_H
        
        win = curses.newwin(height, w, start_row, 0)
        win.box()
        win.attron(curses.A_BOLD)
        win.addstr(0, 2, f" TOTALS ({self.current_filter}) ")
        win.attroff(curses.A_BOLD)
        
        label_col_width = self.col_widths[0] + 2 + self.col_widths[1] if self.show_models else self.col_widths[0]
        
        # Robust column indexing based on total number of columns
        def format_total_line(label: str, stats: Dict[str, Any]) -> str:
            num_cols = len(self.col_widths)
            # Input, Cached, Output, Total, Cost are the last 5 columns
            cost_idx = num_cols - 1
            total_idx = num_cols - 2
            out_idx = num_cols - 3
            cached_idx = num_cols - 4
            input_idx = num_cols - 5

            parts = [f"{label:<{label_col_width}}"]
            if self.show_models:
                # Skip Model column (idx 1) and Sessions column (idx 2)
                parts.append(f"{'':>{self.col_widths[2]}}") 
            
            t_in, t_ca, t_out = stats["input"], stats["cached"], stats["output"]
            parts.append(f"{t_in:>{self.col_widths[input_idx]},}")
            parts.append(f"{t_ca:>{self.col_widths[cached_idx]},}")
            parts.append(f"{t_out:>{self.col_widths[out_idx]},}")
            parts.append(f"{(t_in + t_ca + t_out):>{self.col_widths[total_idx]},}")
            
            cost_str = f"${stats['cost']:,.2f}"
            parts.append(f"{cost_str:>{self.col_widths[cost_idx]}}")
            return "  ".join(parts)

        row_idx = 1
        if self.show_models:
            sorted_models = sorted(self.model_totals.keys())
            for model in sorted_models:
                if row_idx >= height - 2:
                    break
                line = format_total_line(f"TOTAL ({model[:30]})", self.model_totals[model])
                win.addstr(row_idx, 1, line[:w-2])
                row_idx += 1
            if row_idx < height - 1:
                win.addstr(row_idx, 1, ("-" * (w - 2))[:w-2])
                row_idx += 1

        if row_idx < height - 1:
            line = format_total_line("GRAND TOTAL (ALL)", self.totals)
            win.attron(curses.A_BOLD)
            win.addstr(row_idx, 1, line[:w-2])
            win.attroff(curses.A_BOLD)
        
        win.refresh()

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
            stdscr.addstr(h - 1, 0, footer.ljust(w)[:w-1])
        except curses.error:
            pass
        stdscr.attroff(curses.A_REVERSE)

    def edit_pricing(self, stdscr: Any) -> None:
        """Launches the system editor to modify pricing config."""
        pricing_path = Path.home() / ".gemini" / "pricing.json"
        pricing_path.parent.mkdir(parents=True, exist_ok=True)
        if not pricing_path.exists():
            with pricing_path.open("w", encoding="utf-8") as f:
                json.dump({"flash_patterns": [], "pro_patterns": []}, f, indent=2)
        
        curses.def_shell_mode()
        stdscr.clear()
        stdscr.refresh()
        editor = os.environ.get('EDITOR', 'vi')
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
            elif key == curses.KEY_DOWN:
                self.menu_selected = (self.menu_selected + 1) % len(self.filter_options)
            elif key in [10, 13, curses.KEY_ENTER]:
                self.current_filter = self.filter_options[self.menu_selected]
                self.show_filter_menu = False
                self.selected_row = 0
                self.scroll_y = 0
                self.refresh_view_data()
                self.table_pad = None
            elif key in [27, ord('f'), ord('F')]:
                self.show_filter_menu = False
            return

        if key in [ord('q'), ord('Q')]:
            self.running = False
        elif key in [ord('r'), ord('R')]:
            self.load_data()
            self.table_pad = None
        elif key in [ord('p'), ord('P')]:
            self.edit_pricing(stdscr)
        elif key in [ord('f'), ord('F')]:
            self.show_filter_menu = True
            self.menu_selected = self.filter_options.index(self.current_filter)
        elif key in [ord('m'), ord('M')]:
            self.show_models = not self.show_models
            self.selected_row = 0
            self.scroll_y = 0
            self.refresh_view_data()
            self.table_pad = None
        elif key == curses.KEY_UP:
            self.selected_row = max(0, self.selected_row - 1)
        elif key == curses.KEY_DOWN:
            self.selected_row = min(len(self.view_data) - 1, self.selected_row + 1)
        elif key == curses.KEY_PPAGE:
            self.selected_row = max(0, self.selected_row - 10)
        elif key == curses.KEY_NPAGE:
            self.selected_row = min(len(self.view_data) - 1, self.selected_row + 10)

    def main_loop(self, stdscr: Any) -> None:
        """Core application loop."""
        curses.curs_set(0)
        stdscr.keypad(True)
        stdscr.nodelay(False)
        
        self.load_data()
        self.table_pad = None
        
        while self.running:
            stdscr.erase()
            self.draw_header(stdscr)
            self.draw_footer(stdscr)
            
            h, w = stdscr.getmaxyx()
            
            # 1. Calculate layout
            totals_h = self.MIN_TOTALS_H
            if self.show_models:
                totals_h = min(len(self.model_totals) + 4, h // 3)
            
            table_y_start = 2
            table_y_end = h - totals_h - 2
            table_h = table_y_end - table_y_start + 1
            
            # 2. Draw static table header
            header_cols = (["DATE", "MODEL", "SESS", "INPUT", "CACHED", "OUTPUT", "TOTAL", "COST"] 
                           if self.show_models else ["DATE", "SESS", "INPUT", "CACHED", "OUTPUT", "TOTAL", "COST"])
            col_header = ""
            for i, col in enumerate(header_cols):
                align = "<" if i < (2 if self.show_models else 1) else ">"
                col_header += f"{col:{align}{self.col_widths[i]}}  "
            try:
                stdscr.addstr(self.COL_HEADER_Y, 0, col_header[:w-1])
            except curses.error:
                pass

            # 3. Handle pad creation and data rendering
            if not self.table_pad:
                self.table_pad = curses.newpad(max(len(self.view_data) + 1, 100), 256)
            
            self.table_pad.erase()
            for i, (line, _) in enumerate(self.view_data):
                if i == self.selected_row:
                    self.table_pad.attron(curses.A_REVERSE)
                self.table_pad.addstr(i, 0, line)
                if i == self.selected_row:
                    self.table_pad.attroff(curses.A_REVERSE)
            
            # 4. Sync scrolling
            if self.selected_row < self.scroll_y:
                self.scroll_y = self.selected_row
            elif self.selected_row >= self.scroll_y + table_h:
                self.scroll_y = self.selected_row - table_h + 1

            # 5. Refresh screen
            stdscr.noutrefresh()
            if table_h > 0:
                self.table_pad.noutrefresh(self.scroll_y, 0, table_y_start, 0, table_y_end, w - 1)
            
            self.draw_totals(stdscr, h - totals_h - 1, totals_h)
            
            if self.show_filter_menu:
                self.draw_filter_menu(stdscr)
            
            curses.doupdate()
            
            # 6. Process input
            self.handle_input(stdscr.getch(), stdscr)


def main() -> None:
    """TUI Entry point."""
    tui = UsageTUI()
    curses.wrapper(tui.main_loop)


if __name__ == "__main__":
    main()
