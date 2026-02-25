"""
Professional Trade Journal Report Generator
Generates CSV, XLSX (Excel with charts/formulas/styling), and PDF (A4) reports
from bot trade history data.

Usage:
    python scripts/generate_trade_report.py [--days 90] [--output reports/] [--format all] [--balance 100]
"""

import argparse
import csv
import json
import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

# ─── Project path setup ────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.trade_analyzer import TradeAnalyzer

# ─── Optional imports (graceful degradation) ───────────────────────────────
try:
    import xlsxwriter
    HAS_XLSXWRITER = True
except ImportError:
    HAS_XLSXWRITER = False

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ─── Constants ─────────────────────────────────────────────────────────────
# Color palette
NAVY = "#1B2A4A"
BLUE = "#2D5F8A"
GOLD = "#E8A838"
GREEN = "#28A745"
RED = "#DC3545"
LIGHT_GRAY = "#F8F9FA"
DARK_GRAY = "#6C757D"
WHITE = "#FFFFFF"

# Excel hex colors (no #)
XL_NAVY = "1B2A4A"
XL_BLUE = "2D5F8A"
XL_GOLD = "E8A838"
XL_GREEN = "28A745"
XL_RED = "DC3545"
XL_LIGHT_GRAY = "F8F9FA"
XL_ALT_ROW = "EDF2F7"
XL_WHITE = "FFFFFF"

BOT_NAME = "XAUUSD SMC Trading Bot"
BOT_VERSION = "v4.0.0"


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def format_duration(minutes) -> str:
    """Format minutes to human-readable duration."""
    if minutes is None:
        return "-"
    minutes = int(round(minutes))
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    mins = minutes % 60
    if hours < 24:
        return f"{hours}h {mins}m"
    days = hours // 24
    hours = hours % 24
    return f"{days}d {hours}h"


def safe_pf(value: float) -> str:
    """Format profit factor, handling infinity."""
    if value == float("inf"):
        return "N/A"
    return f"{value:.2f}"


# ═══════════════════════════════════════════════════════════════════════════
# ChartGenerator — matplotlib charts as PNG
# ═══════════════════════════════════════════════════════════════════════════

class ChartGenerator:
    """Generate matplotlib charts as PNG files for embedding in reports."""

    def __init__(self, chart_dir: str):
        self.chart_dir = Path(chart_dir)
        self.chart_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = 150

        # Style defaults
        plt.rcParams.update({
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "axes.edgecolor": DARK_GRAY,
            "axes.labelcolor": NAVY,
            "xtick.color": DARK_GRAY,
            "ytick.color": DARK_GRAY,
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
        })

    def equity_curve(self, equities: List[float], dates: List[str],
                     initial_balance: float = 100.0) -> str:
        """Line chart with fill-under for equity curve."""
        path = str(self.chart_dir / "equity_curve.png")
        fig, ax = plt.subplots(figsize=(10, 4))

        # Prepend starting balance as trade #0 for context
        plot_eq = [initial_balance] + list(equities)
        x = range(len(plot_eq))
        ax.plot(x, plot_eq, color=BLUE, linewidth=1.5, label="Equity")
        ax.fill_between(x, plot_eq, alpha=0.15, color=BLUE)

        ax.axhline(y=initial_balance, color=GOLD, linestyle="--", linewidth=0.8,
                   alpha=0.7, label=f"Starting: ${initial_balance:.2f}")

        ax.set_title("Equity Curve")
        ax.set_ylabel("Equity ($)")
        ax.set_xlabel("Trade #")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def drawdown_curve(self, drawdowns: List[float]) -> str:
        """Red area chart for drawdown percentage."""
        path = str(self.chart_dir / "drawdown.png")
        fig, ax = plt.subplots(figsize=(10, 3))

        x = range(len(drawdowns))
        ax.fill_between(x, drawdowns, color=RED, alpha=0.3)
        ax.plot(x, drawdowns, color=RED, linewidth=1.0)

        ax.set_title("Drawdown (%)")
        ax.set_ylabel("Drawdown %")
        ax.set_xlabel("Trade #")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))

        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def daily_pnl_bars(self, daily_data: Dict[str, float]) -> str:
        """Green/red column chart for daily P&L."""
        path = str(self.chart_dir / "daily_pnl.png")
        fig, ax = plt.subplots(figsize=(10, 4))

        dates = list(daily_data.keys())
        pnls = list(daily_data.values())
        colors = [GREEN if p >= 0 else RED for p in pnls]

        ax.bar(range(len(dates)), pnls, color=colors, alpha=0.8, width=0.7)

        ax.set_title("Daily P&L")
        ax.set_ylabel("P&L ($)")
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=7)
        ax.axhline(y=0, color=DARK_GRAY, linewidth=0.5)
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def win_loss_pie(self, wins: int, losses: int, breakevens: int) -> str:
        """3-segment pie chart: W/L/BE."""
        path = str(self.chart_dir / "win_loss_pie.png")
        fig, ax = plt.subplots(figsize=(5, 5))

        labels, sizes, colors = [], [], []
        if wins > 0:
            labels.append(f"Wins ({wins})")
            sizes.append(wins)
            colors.append(GREEN)
        if losses > 0:
            labels.append(f"Losses ({losses})")
            sizes.append(losses)
            colors.append(RED)
        if breakevens > 0:
            labels.append(f"BE ({breakevens})")
            sizes.append(breakevens)
            colors.append(GOLD)

        if not sizes:
            labels, sizes, colors = ["No Trades"], [1], [LIGHT_GRAY]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 9}
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_color(WHITE)
            at.set_fontweight("bold")

        ax.set_title("Win/Loss Distribution")

        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def session_pie(self, session_data: Dict[str, Dict]) -> str:
        """Pie chart by trading session."""
        path = str(self.chart_dir / "session_pie.png")
        fig, ax = plt.subplots(figsize=(5, 5))

        labels = list(session_data.keys())
        sizes = [v["total"] for v in session_data.values()]
        color_cycle = [BLUE, GOLD, GREEN, RED, DARK_GRAY, NAVY]

        if not sizes:
            labels, sizes = ["No Data"], [1]
            color_cycle = [LIGHT_GRAY]

        ax.pie(sizes, labels=labels,
               colors=color_cycle[:len(labels)],
               autopct="%1.1f%%", startangle=90,
               textprops={"fontsize": 9})
        ax.set_title("Trades by Session")

        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def regime_pie(self, regime_data: Dict[str, Dict]) -> str:
        """Pie chart by market regime. Skip if only 1 segment or all Unknown."""
        path = str(self.chart_dir / "regime_pie.png")
        fig, ax = plt.subplots(figsize=(5, 5))

        # Filter out "Unknown" if there are real regimes
        filtered = {k: v for k, v in regime_data.items() if k != "Unknown"}
        if not filtered:
            filtered = regime_data  # fall back to showing Unknown

        labels = list(filtered.keys())
        sizes = [v["total"] for v in filtered.values()]
        color_cycle = [NAVY, BLUE, GOLD, GREEN, RED, DARK_GRAY, "#8B5CF6", "#06B6D4"]

        if not sizes or (len(sizes) == 1 and labels[0] == "Unknown"):
            ax.axis("off")
            ax.text(0.5, 0.5, "No regime data\n(regime not recorded)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color=DARK_GRAY)
            ax.set_title("Trades by Regime")
        else:
            ax.pie(sizes, labels=labels,
                   colors=color_cycle[:len(labels)],
                   autopct="%1.1f%%", startangle=90,
                   textprops={"fontsize": 8})
            ax.set_title("Trades by Regime")

        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def hourly_performance(self, hourly_data: Dict[str, Dict]) -> str:
        """Bar chart by UTC hour."""
        path = str(self.chart_dir / "hourly_perf.png")
        fig, ax = plt.subplots(figsize=(10, 4))

        hours = list(hourly_data.keys())
        pnls = [v["net_profit"] for v in hourly_data.values()]
        colors = [GREEN if p >= 0 else RED for p in pnls]

        labels = [f"{h}:00" if not str(h).endswith(":00") else str(h) for h in hours]
        ax.bar(range(len(hours)), pnls, color=colors, alpha=0.8, width=0.7)
        ax.set_title("P&L by Hour (UTC)")
        ax.set_ylabel("Net P&L ($)")
        ax.set_xticks(range(len(hours)))
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
        ax.axhline(y=0, color=DARK_GRAY, linewidth=0.5)
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def rr_distribution(self, rr_values: List[float]) -> str:
        """Histogram of R:R distribution."""
        path = str(self.chart_dir / "rr_dist.png")
        fig, ax = plt.subplots(figsize=(8, 4))

        if rr_values:
            n, bins, patches = ax.hist(rr_values, bins=20, color=BLUE, alpha=0.7,
                                       edgecolor=WHITE, linewidth=0.5)
            for patch, left_edge in zip(patches, bins):
                if left_edge >= 0:
                    patch.set_facecolor(GREEN)
                else:
                    patch.set_facecolor(RED)
                patch.set_alpha(0.7)
        else:
            ax.text(0.5, 0.5, "No R:R data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)

        ax.set_title("R:R Distribution")
        ax.set_xlabel("Risk:Reward Ratio")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.axvline(x=0, color=DARK_GRAY, linewidth=0.8, linestyle="--")
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path


# ═══════════════════════════════════════════════════════════════════════════
# ExcelReportBuilder — xlsxwriter with 9 sheets
# ═══════════════════════════════════════════════════════════════════════════

class ExcelReportBuilder:
    """Build Excel report with 9 sheets, charts, formulas, and styling."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.wb = xlsxwriter.Workbook(filepath)
        self._setup_formats()

    def _setup_formats(self):
        """Create reusable cell formats."""
        self.fmt_title = self.wb.add_format({
            "font_name": "Calibri", "font_size": 16, "bold": True,
            "font_color": XL_WHITE, "bg_color": XL_NAVY,
            "align": "center", "valign": "vcenter",
            "border": 1, "border_color": XL_NAVY,
        })
        self.fmt_header = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10, "bold": True,
            "font_color": XL_WHITE, "bg_color": XL_BLUE,
            "align": "center", "valign": "vcenter",
            "border": 1, "text_wrap": True,
        })
        self.fmt_subheader = self.wb.add_format({
            "font_name": "Calibri", "font_size": 11, "bold": True,
            "font_color": XL_WHITE, "bg_color": XL_BLUE,
            "align": "left", "valign": "vcenter",
            "border": 1,
        })
        self.fmt_body = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10,
            "align": "center", "valign": "vcenter",
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_body_left = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10,
            "align": "left", "valign": "vcenter",
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_alt_row = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10,
            "align": "center", "valign": "vcenter",
            "bg_color": XL_ALT_ROW,
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_alt_row_left = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10,
            "align": "left", "valign": "vcenter",
            "bg_color": XL_ALT_ROW,
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_currency = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10,
            "num_format": "$#,##0.00", "align": "center",
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_currency_alt = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10,
            "num_format": "$#,##0.00", "align": "center",
            "bg_color": XL_ALT_ROW,
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_pct = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10,
            "num_format": "0.0%", "align": "center",
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_pct_alt = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10,
            "num_format": "0.0%", "align": "center",
            "bg_color": XL_ALT_ROW,
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_decimal = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10,
            "num_format": "0.00", "align": "center",
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_decimal_alt = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10,
            "num_format": "0.00", "align": "center",
            "bg_color": XL_ALT_ROW,
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_green = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10,
            "num_format": "$#,##0.00", "align": "center",
            "font_color": XL_GREEN, "bold": True,
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_red = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10,
            "num_format": "$#,##0.00", "align": "center",
            "font_color": XL_RED, "bold": True,
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_green_alt = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10,
            "num_format": "$#,##0.00", "align": "center",
            "font_color": XL_GREEN, "bold": True,
            "bg_color": XL_ALT_ROW,
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_red_alt = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10,
            "num_format": "$#,##0.00", "align": "center",
            "font_color": XL_RED, "bold": True,
            "bg_color": XL_ALT_ROW,
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_gold_row = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10, "bold": True,
            "bg_color": XL_GOLD, "font_color": XL_NAVY,
            "align": "center", "border": 2,
        })
        self.fmt_gold_currency = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10, "bold": True,
            "bg_color": XL_GOLD, "font_color": XL_NAVY,
            "num_format": "$#,##0.00", "align": "center", "border": 2,
        })
        self.fmt_gold_pct = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10, "bold": True,
            "bg_color": XL_GOLD, "font_color": XL_NAVY,
            "num_format": "0.0%", "align": "center", "border": 2,
        })
        self.fmt_ticket = self.wb.add_format({
            "font_name": "Consolas", "font_size": 10,
            "align": "center",
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_ticket_alt = self.wb.add_format({
            "font_name": "Consolas", "font_size": 10,
            "align": "center", "bg_color": XL_ALT_ROW,
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_kpi_label = self.wb.add_format({
            "font_name": "Calibri", "font_size": 10, "bold": True,
            "font_color": XL_NAVY, "align": "right", "valign": "vcenter",
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_kpi_value = self.wb.add_format({
            "font_name": "Calibri", "font_size": 12, "bold": True,
            "font_color": XL_BLUE, "align": "left", "valign": "vcenter",
            "border": 1, "border_color": "D0D0D0",
        })
        self.fmt_kpi_currency = self.wb.add_format({
            "font_name": "Calibri", "font_size": 12, "bold": True,
            "font_color": XL_BLUE, "num_format": "$#,##0.00",
            "align": "left", "valign": "vcenter",
            "border": 1, "border_color": "D0D0D0",
        })

    def _write_title_row(self, ws, row: int, title: str, col_span: int) -> int:
        """Write a merged title row."""
        ws.merge_range(row, 0, row, col_span - 1, title, self.fmt_title)
        ws.set_row(row, 30)
        return row + 1

    def _write_headers(self, ws, row: int, headers: List[str]) -> int:
        """Write header row with formatting."""
        for col, h in enumerate(headers):
            ws.write(row, col, h, self.fmt_header)
        ws.set_row(row, 22)
        return row + 1

    def _get_row_fmt(self, row_idx: int, fmt_normal, fmt_alt):
        """Return alternating row format."""
        return fmt_alt if row_idx % 2 == 1 else fmt_normal

    def build(self, trades_df: List[Dict], analysis: Dict, initial_balance: float,
              daily_agg: List[Dict], weekly_agg: List[Dict], monthly_agg: List[Dict]):
        """Build all 9 sheets."""
        self._sheet_dashboard(trades_df, analysis, initial_balance)
        self._sheet_all_trades(trades_df)
        self._sheet_period_agg("Daily", daily_agg)
        self._sheet_period_agg("Weekly", weekly_agg)
        self._sheet_period_agg("Monthly", monthly_agg)
        self._sheet_session(analysis.get("by_session", {}))
        self._sheet_smc(analysis.get("by_smc_combo", {}))
        self._sheet_regime(analysis.get("by_regime", {}))
        self._sheet_charts(trades_df, analysis, daily_agg)
        self.wb.close()

    def _sheet_dashboard(self, trades: List[Dict], analysis: Dict, balance: float):
        """Dashboard sheet with KPI grid and mini tables."""
        ws = self.wb.add_worksheet("Dashboard")
        ws.hide_gridlines(2)
        ws.set_column("A:A", 22)
        ws.set_column("B:B", 16)
        ws.set_column("C:C", 5)
        ws.set_column("D:D", 22)
        ws.set_column("E:E", 16)
        ws.set_column("F:F", 5)
        ws.set_column("G:G", 22)
        ws.set_column("H:H", 16)

        row = 0
        ws.merge_range(row, 0, row, 7, f"{BOT_NAME} {BOT_VERSION} - Trade Report", self.fmt_title)
        ws.set_row(row, 35)
        row += 1

        now = datetime.now(tz=None).strftime("%Y-%m-%d %H:%M UTC")
        ws.merge_range(row, 0, row, 7, f"Generated: {now}", self.wb.add_format({
            "font_name": "Calibri", "font_size": 9, "italic": True,
            "font_color": XL_BLUE, "align": "center",
        }))
        row += 2

        overall = analysis.get("overall", {})

        # KPI Grid — 4 rows x 3 columns (label+value pairs)
        kpis = [
            # Row 1
            ("Total Trades", str(overall.get("total_trades", 0))),
            ("Win Rate", f"{overall.get('win_rate', 0):.1%}"),
            ("Net P&L", f"${overall.get('net_profit', 0):.2f}"),
            # Row 2
            ("Wins / Losses / BE",
             f"{overall.get('wins', 0)} / {overall.get('losses', 0)} / {overall.get('breakevens', 0)}"),
            ("Profit Factor", safe_pf(overall.get("profit_factor", 0))),
            ("Starting Balance", f"${balance:.2f}"),
            # Row 3
            ("Avg Win", f"${overall.get('avg_win', 0):.2f}"),
            ("Avg Loss", f"${abs(overall.get('avg_loss', 0)):.2f}"),
            ("Avg Duration",
             format_duration(overall.get("avg_duration_minutes"))),
            # Row 4
            ("Max Consec Losses", str(overall.get("max_consecutive_losses", 0))),
            ("Best Trade",
             f"${overall['best_trade']['profit']:.2f}" if overall.get("best_trade") else "-"),
            ("Worst Trade",
             f"${overall['worst_trade']['profit']:.2f}" if overall.get("worst_trade") else "-"),
        ]

        ws.merge_range(row, 0, row, 7, "KEY PERFORMANCE INDICATORS", self.fmt_subheader)
        ws.set_row(row, 22)
        row += 1

        for i in range(0, len(kpis), 3):
            chunk = kpis[i:i+3]
            for j, (label, value) in enumerate(chunk):
                col_label = j * 3
                col_value = col_label + 1
                ws.write(row, col_label, label, self.fmt_kpi_label)
                ws.write(row, col_value, value, self.fmt_kpi_value)
            ws.set_row(row, 22)
            row += 1

        row += 1

        # Mini session table
        by_session = analysis.get("by_session", {})
        if by_session:
            ws.merge_range(row, 0, row, 4, "SESSION BREAKDOWN", self.fmt_subheader)
            row += 1
            mini_headers = ["Session", "Trades", "Win Rate", "PF", "Net P&L"]
            for c, h in enumerate(mini_headers):
                ws.write(row, c, h, self.fmt_header)
            row += 1
            for idx, (name, s) in enumerate(by_session.items()):
                f_body = self._get_row_fmt(idx, self.fmt_body, self.fmt_alt_row)
                f_left = self._get_row_fmt(idx, self.fmt_body_left, self.fmt_alt_row_left)
                f_pct = self._get_row_fmt(idx, self.fmt_pct, self.fmt_pct_alt)
                f_dec = self._get_row_fmt(idx, self.fmt_decimal, self.fmt_decimal_alt)
                f_cur = self._get_row_fmt(idx, self.fmt_currency, self.fmt_currency_alt)
                ws.write(row, 0, name, f_left)
                ws.write(row, 1, s["total"], f_body)
                ws.write(row, 2, s["win_rate"], f_pct)
                ws.write(row, 3, s["profit_factor"] if s["profit_factor"] != float("inf") else 999, f_dec)
                ws.write(row, 4, s["net_profit"], f_cur)
                row += 1
            row += 1

        # Mini regime table
        by_regime = analysis.get("by_regime", {})
        if by_regime:
            ws.merge_range(row, 0, row, 4, "REGIME BREAKDOWN", self.fmt_subheader)
            row += 1
            mini_headers = ["Regime", "Trades", "Win Rate", "PF", "Net P&L"]
            for c, h in enumerate(mini_headers):
                ws.write(row, c, h, self.fmt_header)
            row += 1
            for idx, (name, s) in enumerate(by_regime.items()):
                f_body = self._get_row_fmt(idx, self.fmt_body, self.fmt_alt_row)
                f_left = self._get_row_fmt(idx, self.fmt_body_left, self.fmt_alt_row_left)
                f_pct = self._get_row_fmt(idx, self.fmt_pct, self.fmt_pct_alt)
                f_dec = self._get_row_fmt(idx, self.fmt_decimal, self.fmt_decimal_alt)
                f_cur = self._get_row_fmt(idx, self.fmt_currency, self.fmt_currency_alt)
                ws.write(row, 0, name, f_left)
                ws.write(row, 1, s["total"], f_body)
                ws.write(row, 2, s["win_rate"], f_pct)
                ws.write(row, 3, s["profit_factor"] if s["profit_factor"] != float("inf") else 999, f_dec)
                ws.write(row, 4, s["net_profit"], f_cur)
                row += 1

    def _sheet_all_trades(self, trades: List[Dict]):
        """Full trade log with formulas and conditional formatting."""
        ws = self.wb.add_worksheet("All Trades")
        ws.freeze_panes(2, 0)

        headers = [
            "#", "Ticket", "Dir", "Entry Date", "Entry Time",
            "Exit Date", "Exit Time", "Entry Price", "Exit Price",
            "SL", "TP", "SL Pips", "TP Pips", "Volume",
            "P&L ($)", "R:R", "Duration", "Confluence",
            "SMC Combo", "Session", "Regime", "Exit Reason",
            "MFE ($)", "MAE ($)", "Stage", "W/L",
            "Cum P&L ($)", "Equity ($)", "DD %",
        ]

        col_widths = [
            5, 14, 5, 11, 9, 11, 9, 10, 10,
            10, 10, 7, 7, 7,
            9, 6, 9, 8,
            16, 16, 14, 22,
            9, 9, 10, 5,
            10, 10, 7,
        ]

        row = self._write_title_row(ws, 0, "Complete Trade Log", len(headers))
        row = self._write_headers(ws, row, headers)

        for i, w in enumerate(col_widths):
            ws.set_column(i, i, w)

        if not trades:
            ws.merge_range(row, 0, row, len(headers) - 1, "No trades to display",
                          self.fmt_body)
            return

        pnl_col = 14  # P&L column index (0-based)

        for idx, t in enumerate(trades):
            f_body = self._get_row_fmt(idx, self.fmt_body, self.fmt_alt_row)
            f_left = self._get_row_fmt(idx, self.fmt_body_left, self.fmt_alt_row_left)
            f_ticket = self._get_row_fmt(idx, self.fmt_ticket, self.fmt_ticket_alt)
            f_dec = self._get_row_fmt(idx, self.fmt_decimal, self.fmt_decimal_alt)
            f_pct = self._get_row_fmt(idx, self.fmt_pct, self.fmt_pct_alt)

            pnl = t.get("profit_usd", 0)
            if pnl >= 0:
                f_pnl = self._get_row_fmt(idx, self.fmt_green, self.fmt_green_alt)
            else:
                f_pnl = self._get_row_fmt(idx, self.fmt_red, self.fmt_red_alt)

            c = 0
            ws.write_number(row, c, t.get("trade_no", idx + 1), f_body); c += 1
            ws.write_string(row, c, str(t.get("ticket", "")), f_ticket); c += 1
            ws.write_string(row, c, t.get("direction", ""), f_body); c += 1
            ws.write_string(row, c, t.get("entry_date", ""), f_body); c += 1
            ws.write_string(row, c, t.get("entry_time_utc", ""), f_body); c += 1
            ws.write_string(row, c, t.get("exit_date", ""), f_body); c += 1
            ws.write_string(row, c, t.get("exit_time_utc", ""), f_body); c += 1
            ws.write_number(row, c, t.get("entry_price", 0), f_dec); c += 1
            ws.write_number(row, c, t.get("exit_price", 0), f_dec); c += 1
            ws.write_number(row, c, t.get("sl", 0), f_dec); c += 1
            ws.write_number(row, c, t.get("tp", 0), f_dec); c += 1
            ws.write_number(row, c, t.get("sl_pips", 0), f_dec); c += 1
            ws.write_number(row, c, t.get("tp_pips", 0), f_dec); c += 1
            ws.write_number(row, c, t.get("volume", 0.01), f_dec); c += 1
            ws.write_number(row, c, pnl, f_pnl); c += 1
            ws.write_number(row, c, t.get("rr_ratio", 0), f_dec); c += 1
            ws.write_string(row, c, t.get("duration_str", ""), f_body); c += 1
            ws.write_number(row, c, t.get("confluence", 0), f_dec); c += 1
            ws.write_string(row, c, t.get("smc_combo", ""), f_left); c += 1
            ws.write_string(row, c, t.get("session", ""), f_left); c += 1
            ws.write_string(row, c, t.get("regime", ""), f_left); c += 1
            ws.write_string(row, c, t.get("exit_reason", ""), f_left); c += 1
            ws.write_number(row, c, t.get("mfe_usd", 0), f_dec); c += 1
            ws.write_number(row, c, t.get("mae_usd", 0), f_dec); c += 1
            ws.write_string(row, c, t.get("stage_reached", ""), f_body); c += 1
            ws.write_string(row, c, "W" if t.get("is_win") else ("L" if t.get("is_loss") else "BE"), f_body); c += 1
            ws.write_number(row, c, t.get("cumulative_pnl", 0), f_pnl); c += 1
            ws.write_number(row, c, t.get("equity", 0), f_dec); c += 1
            ws.write_number(row, c, t.get("drawdown_pct", 0) / 100 if t.get("drawdown_pct") else 0, f_pct); c += 1
            row += 1

        # Summary row
        data_start = 2  # header is row 1 (0-indexed)
        data_end = row - 1
        sr = row

        ws.write(sr, 0, "TOTAL", self.fmt_gold_row)
        for c in range(1, pnl_col):
            ws.write_blank(sr, c, None, self.fmt_gold_row)

        # SUM of P&L
        pnl_cell_range = xlsxwriter.utility.xl_range(data_start, pnl_col, data_end, pnl_col)
        ws.write_formula(sr, pnl_col, f"=SUM({pnl_cell_range})", self.fmt_gold_currency)

        for c in range(pnl_col + 1, len(headers)):
            ws.write_blank(sr, c, None, self.fmt_gold_row)

        # Auto-filter
        ws.autofilter(1, 0, data_end, len(headers) - 1)

    def _sheet_period_agg(self, period_name: str, agg_data: List[Dict]):
        """Daily/Weekly/Monthly aggregation sheet."""
        ws = self.wb.add_worksheet(period_name)
        ws.freeze_panes(2, 0)

        headers = ["Period", "Trades", "Wins", "Losses", "Win Rate",
                    "PF", "Net P&L ($)", "Cumulative ($)"]

        col_widths = [14, 8, 8, 8, 10, 8, 12, 12]

        row = self._write_title_row(ws, 0, f"{period_name} Performance", len(headers))
        row = self._write_headers(ws, row, headers)

        for i, w in enumerate(col_widths):
            ws.set_column(i, i, w)

        if not agg_data:
            ws.merge_range(row, 0, row, len(headers) - 1, "No data", self.fmt_body)
            return

        for idx, d in enumerate(agg_data):
            f_body = self._get_row_fmt(idx, self.fmt_body, self.fmt_alt_row)
            f_left = self._get_row_fmt(idx, self.fmt_body_left, self.fmt_alt_row_left)
            f_pct = self._get_row_fmt(idx, self.fmt_pct, self.fmt_pct_alt)
            f_dec = self._get_row_fmt(idx, self.fmt_decimal, self.fmt_decimal_alt)

            pnl = d.get("net_pnl", 0)
            f_pnl = self._get_row_fmt(idx, self.fmt_green if pnl >= 0 else self.fmt_red,
                                       self.fmt_green_alt if pnl >= 0 else self.fmt_red_alt)

            ws.write_string(row, 0, d.get("period", ""), f_left)
            ws.write_number(row, 1, d.get("trades", 0), f_body)
            ws.write_number(row, 2, d.get("wins", 0), f_body)
            ws.write_number(row, 3, d.get("losses", 0), f_body)
            ws.write_number(row, 4, d.get("win_rate", 0), f_pct)
            ws.write_number(row, 5,
                           d.get("pf", 0) if d.get("pf", 0) != float("inf") else 999,
                           f_dec)
            ws.write_number(row, 6, pnl, f_pnl)
            cum = d.get("cumulative", 0)
            f_cum = self._get_row_fmt(idx, self.fmt_green if cum >= 0 else self.fmt_red,
                                       self.fmt_green_alt if cum >= 0 else self.fmt_red_alt)
            ws.write_number(row, 7, cum, f_cum)
            row += 1

        # Summary row
        sr = row
        ws.write(sr, 0, "TOTAL", self.fmt_gold_row)
        total_trades = sum(d.get("trades", 0) for d in agg_data)
        total_wins = sum(d.get("wins", 0) for d in agg_data)
        total_losses = sum(d.get("losses", 0) for d in agg_data)
        total_pnl = sum(d.get("net_pnl", 0) for d in agg_data)
        ws.write_number(sr, 1, total_trades, self.fmt_gold_row)
        ws.write_number(sr, 2, total_wins, self.fmt_gold_row)
        ws.write_number(sr, 3, total_losses, self.fmt_gold_row)
        ws.write_number(sr, 4, total_wins / total_trades if total_trades else 0, self.fmt_gold_pct)
        ws.write(sr, 5, "", self.fmt_gold_row)
        ws.write_number(sr, 6, total_pnl, self.fmt_gold_currency)
        ws.write(sr, 7, "", self.fmt_gold_row)

    def _sheet_session(self, session_data: Dict[str, Dict]):
        """Session analysis sheet."""
        ws = self.wb.add_worksheet("Session Analysis")
        ws.freeze_panes(2, 0)

        headers = ["Session", "Trades", "Wins", "Losses", "Win Rate",
                    "PF", "Net P&L ($)", "Avg Win ($)", "Avg Loss ($)"]

        col_widths = [18, 8, 8, 8, 10, 8, 12, 12, 12]

        row = self._write_title_row(ws, 0, "Session Analysis", len(headers))
        row = self._write_headers(ws, row, headers)

        for i, w in enumerate(col_widths):
            ws.set_column(i, i, w)

        if not session_data:
            ws.merge_range(row, 0, row, len(headers) - 1, "No data", self.fmt_body)
            return

        for idx, (name, s) in enumerate(session_data.items()):
            f_body = self._get_row_fmt(idx, self.fmt_body, self.fmt_alt_row)
            f_left = self._get_row_fmt(idx, self.fmt_body_left, self.fmt_alt_row_left)
            f_pct = self._get_row_fmt(idx, self.fmt_pct, self.fmt_pct_alt)
            f_dec = self._get_row_fmt(idx, self.fmt_decimal, self.fmt_decimal_alt)
            f_cur = self._get_row_fmt(idx, self.fmt_currency, self.fmt_currency_alt)

            ws.write_string(row, 0, name, f_left)
            ws.write_number(row, 1, s["total"], f_body)
            ws.write_number(row, 2, s["wins"], f_body)
            ws.write_number(row, 3, s["losses"], f_body)
            ws.write_number(row, 4, s["win_rate"], f_pct)
            pf = s["profit_factor"]
            ws.write_number(row, 5, pf if pf != float("inf") else 999, f_dec)
            ws.write_number(row, 6, s["net_profit"], f_cur)
            ws.write_number(row, 7, s["avg_win"], f_cur)
            ws.write_number(row, 8, s["avg_loss"], f_cur)
            row += 1

    def _sheet_smc(self, smc_data: Dict[str, Dict]):
        """SMC signal combination performance."""
        ws = self.wb.add_worksheet("SMC Performance")
        ws.freeze_panes(2, 0)

        headers = ["SMC Combo", "Trades", "Wins", "Win Rate",
                    "Net P&L ($)", "Avg P&L ($)"]

        col_widths = [22, 8, 8, 10, 12, 12]

        row = self._write_title_row(ws, 0, "SMC Signal Combo Performance", len(headers))
        row = self._write_headers(ws, row, headers)

        for i, w in enumerate(col_widths):
            ws.set_column(i, i, w)

        if not smc_data:
            ws.merge_range(row, 0, row, len(headers) - 1, "No data", self.fmt_body)
            return

        for idx, (combo, s) in enumerate(smc_data.items()):
            f_body = self._get_row_fmt(idx, self.fmt_body, self.fmt_alt_row)
            f_left = self._get_row_fmt(idx, self.fmt_body_left, self.fmt_alt_row_left)
            f_pct = self._get_row_fmt(idx, self.fmt_pct, self.fmt_pct_alt)
            f_cur = self._get_row_fmt(idx, self.fmt_currency, self.fmt_currency_alt)

            ws.write_string(row, 0, combo, f_left)
            ws.write_number(row, 1, s["total"], f_body)
            ws.write_number(row, 2, s["wins"], f_body)
            ws.write_number(row, 3, s["win_rate"], f_pct)
            ws.write_number(row, 4, s["net_profit"], f_cur)
            ws.write_number(row, 5, s["avg_profit"], f_cur)
            row += 1

    def _sheet_regime(self, regime_data: Dict[str, Dict]):
        """Market regime analysis."""
        ws = self.wb.add_worksheet("Regime Analysis")
        ws.freeze_panes(2, 0)

        headers = ["Regime", "Trades", "Wins", "Losses", "Win Rate",
                    "PF", "Net P&L ($)"]

        col_widths = [18, 8, 8, 8, 10, 8, 12]

        row = self._write_title_row(ws, 0, "Market Regime Analysis", len(headers))
        row = self._write_headers(ws, row, headers)

        for i, w in enumerate(col_widths):
            ws.set_column(i, i, w)

        if not regime_data:
            ws.merge_range(row, 0, row, len(headers) - 1, "No data", self.fmt_body)
            return

        for idx, (name, s) in enumerate(regime_data.items()):
            f_body = self._get_row_fmt(idx, self.fmt_body, self.fmt_alt_row)
            f_left = self._get_row_fmt(idx, self.fmt_body_left, self.fmt_alt_row_left)
            f_pct = self._get_row_fmt(idx, self.fmt_pct, self.fmt_pct_alt)
            f_dec = self._get_row_fmt(idx, self.fmt_decimal, self.fmt_decimal_alt)
            f_cur = self._get_row_fmt(idx, self.fmt_currency, self.fmt_currency_alt)

            ws.write_string(row, 0, name, f_left)
            ws.write_number(row, 1, s["total"], f_body)
            ws.write_number(row, 2, s["wins"], f_body)
            ws.write_number(row, 3, s["losses"], f_body)
            ws.write_number(row, 4, s["win_rate"], f_pct)
            pf = s["profit_factor"]
            ws.write_number(row, 5, pf if pf != float("inf") else 999, f_dec)
            ws.write_number(row, 6, s["net_profit"], f_cur)
            row += 1

    def _sheet_charts(self, trades: List[Dict], analysis: Dict, daily_agg: List[Dict]):
        """Charts sheet with native xlsxwriter charts."""
        ws = self.wb.add_worksheet("Charts")
        ws.hide_gridlines(2)

        if not trades:
            ws.write(0, 0, "No trade data for charts", self.fmt_body)
            return

        # Write hidden data for charts
        # Equity data
        data_ws = self.wb.add_worksheet("_ChartData")
        data_ws.hide()

        data_ws.write(0, 0, "Trade #")
        data_ws.write(0, 1, "Equity")
        data_ws.write(0, 2, "Drawdown %")
        data_ws.write(0, 3, "Day")
        data_ws.write(0, 4, "Daily P&L")

        for i, t in enumerate(trades):
            data_ws.write(i + 1, 0, i + 1)
            data_ws.write(i + 1, 1, t.get("equity", 0))
            data_ws.write(i + 1, 2, t.get("drawdown_pct", 0))

        for i, d in enumerate(daily_agg):
            data_ws.write(i + 1, 3, d.get("period", ""))
            data_ws.write(i + 1, 4, d.get("net_pnl", 0))

        n_trades = len(trades)
        n_days = len(daily_agg)

        # Equity curve chart
        chart_eq = self.wb.add_chart({"type": "line"})
        chart_eq.add_series({
            "name": "Equity",
            "categories": ["_ChartData", 1, 0, n_trades, 0],
            "values": ["_ChartData", 1, 1, n_trades, 1],
            "line": {"color": BLUE, "width": 2},
        })
        chart_eq.set_title({"name": "Equity Curve"})
        chart_eq.set_x_axis({"name": "Trade #"})
        chart_eq.set_y_axis({"name": "Equity ($)", "num_format": "$#,##0.00"})
        chart_eq.set_size({"width": 800, "height": 400})
        chart_eq.set_legend({"none": True})
        ws.insert_chart("A1", chart_eq)

        # Drawdown chart
        chart_dd = self.wb.add_chart({"type": "area"})
        chart_dd.add_series({
            "name": "Drawdown %",
            "categories": ["_ChartData", 1, 0, n_trades, 0],
            "values": ["_ChartData", 1, 2, n_trades, 2],
            "fill": {"color": RED, "transparency": 60},
            "line": {"color": RED, "width": 1},
        })
        chart_dd.set_title({"name": "Drawdown (%)"})
        chart_dd.set_x_axis({"name": "Trade #"})
        chart_dd.set_y_axis({"name": "Drawdown %", "num_format": "0.0%"})
        chart_dd.set_size({"width": 800, "height": 300})
        chart_dd.set_legend({"none": True})
        ws.insert_chart("A21", chart_dd)

        # Daily P&L bar chart
        if n_days > 0:
            chart_daily = self.wb.add_chart({"type": "column"})
            chart_daily.add_series({
                "name": "Daily P&L",
                "categories": ["_ChartData", 1, 3, n_days, 3],
                "values": ["_ChartData", 1, 4, n_days, 4],
                "fill": {"color": BLUE},
            })
            chart_daily.set_title({"name": "Daily P&L"})
            chart_daily.set_y_axis({"name": "P&L ($)", "num_format": "$#,##0.00"})
            chart_daily.set_size({"width": 800, "height": 350})
            chart_daily.set_legend({"none": True})
            ws.insert_chart("A37", chart_daily)

        # Win/Loss pie
        overall = analysis.get("overall", {})
        wins = overall.get("wins", 0)
        losses = overall.get("losses", 0)
        bes = overall.get("breakevens", 0)

        data_ws.write(0, 6, "Category")
        data_ws.write(0, 7, "Count")
        pie_labels = ["Wins", "Losses", "BE"]
        pie_vals = [wins, losses, bes]
        for i, (lbl, val) in enumerate(zip(pie_labels, pie_vals)):
            data_ws.write(i + 1, 6, lbl)
            data_ws.write(i + 1, 7, val)

        chart_pie = self.wb.add_chart({"type": "pie"})
        chart_pie.add_series({
            "name": "Win/Loss",
            "categories": ["_ChartData", 1, 6, 3, 6],
            "values": ["_ChartData", 1, 7, 3, 7],
            "points": [
                {"fill": {"color": GREEN}},
                {"fill": {"color": RED}},
                {"fill": {"color": GOLD}},
            ],
        })
        chart_pie.set_title({"name": "Win/Loss Distribution"})
        chart_pie.set_size({"width": 400, "height": 350})
        ws.insert_chart("J1", chart_pie)


# ═══════════════════════════════════════════════════════════════════════════
# PDFReportBuilder — fpdf2 A4 pages with embedded charts
# ═══════════════════════════════════════════════════════════════════════════

class _NumberedPDF(FPDF):
    """FPDF subclass with page numbers and bot name in the footer."""

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        margin = 15
        content_w = self.w - 2 * margin
        # Left: bot name
        self.set_x(margin)
        self.cell(content_w / 2, 5, f"{BOT_NAME} {BOT_VERSION}", align="L")
        # Right: page number
        self.cell(content_w / 2, 5, f"Page {self.page_no()}/{{nb}}", align="R")


class PDFReportBuilder:
    """Build A4 PDF report with tables and embedded matplotlib charts."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.pdf = _NumberedPDF(orientation="P", unit="mm", format="A4")
        self.pdf.alias_nb_pages()
        self.pdf.set_auto_page_break(auto=True, margin=18)
        self.page_w = 210
        self.margin = 15
        self.content_w = self.page_w - 2 * self.margin

    def build(self, trades: List[Dict], analysis: Dict, initial_balance: float,
              daily_agg: List[Dict], chart_paths: Dict[str, str],
              session_data: Dict, regime_data: Dict, smc_data: Dict,
              hourly_data: Dict, days: int = 90):
        """Build complete PDF."""
        self._page_cover(initial_balance, len(trades), days)
        self._page_summary(analysis, chart_paths)
        self._page_trade_log(trades)
        self._page_session(session_data, chart_paths)
        self._page_regime(regime_data, chart_paths)
        self._page_smc(smc_data)
        self._page_hourly(chart_paths)
        self._page_system_config()
        self.pdf.output(self.filepath)

    def _set_navy_header(self, text: str):
        """Add a navy section header (adapts to portrait/landscape)."""
        r, g, b = hex_to_rgb(NAVY)
        self.pdf.set_fill_color(r, g, b)
        self.pdf.set_text_color(255, 255, 255)
        self.pdf.set_font("Helvetica", "B", 12)
        w = self.pdf.epw  # effective page width (adapts to orientation)
        self.pdf.cell(w, 8, text, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(2)

    def _set_blue_subheader(self, text: str):
        """Add a blue sub-header."""
        r, g, b = hex_to_rgb(BLUE)
        self.pdf.set_fill_color(r, g, b)
        self.pdf.set_text_color(255, 255, 255)
        self.pdf.set_font("Helvetica", "B", 10)
        self.pdf.cell(self.content_w, 7, text, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(1)

    def _table_header(self, col_widths: List[float], headers: List[str]):
        """Write table header row."""
        r, g, b = hex_to_rgb(BLUE)
        self.pdf.set_fill_color(r, g, b)
        self.pdf.set_text_color(255, 255, 255)
        self.pdf.set_font("Helvetica", "B", 7)
        for w, h in zip(col_widths, headers):
            self.pdf.cell(w, 6, h, border=1, align="C", fill=True)
        self.pdf.ln()
        self.pdf.set_text_color(0, 0, 0)

    def _table_row(self, col_widths: List[float], values: List[str],
                   aligns: Optional[List[str]] = None, alt: bool = False,
                   pnl_col: Optional[int] = None):
        """Write table data row with optional alternating colors."""
        if alt:
            self.pdf.set_fill_color(237, 242, 247)  # light blue-gray
        else:
            self.pdf.set_fill_color(255, 255, 255)

        self.pdf.set_font("Helvetica", "", 7)

        for i, (w, v) in enumerate(zip(col_widths, values)):
            align = "C"
            if aligns and i < len(aligns):
                align = aligns[i]

            # Color P&L values
            if pnl_col is not None and i == pnl_col:
                try:
                    val = float(v.replace("$", "").replace(",", ""))
                    if val >= 0:
                        self.pdf.set_text_color(*hex_to_rgb(GREEN))
                    else:
                        self.pdf.set_text_color(*hex_to_rgb(RED))
                    self.pdf.set_font("Helvetica", "B", 7)
                except (ValueError, AttributeError):
                    pass

            self.pdf.cell(w, 5, str(v), border=1, align=align, fill=True)

            # Reset
            if pnl_col is not None and i == pnl_col:
                self.pdf.set_text_color(0, 0, 0)
                self.pdf.set_font("Helvetica", "", 7)

        self.pdf.ln()

    def _page_cover(self, balance: float, n_trades: int, days: int = 90):
        """Cover page."""
        self.pdf.add_page()
        self.pdf.ln(40)

        # Title
        r, g, b = hex_to_rgb(NAVY)
        self.pdf.set_text_color(r, g, b)
        self.pdf.set_font("Helvetica", "B", 28)
        self.pdf.cell(self.content_w, 15, "Trade Journal Report", align="C",
                     new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(5)

        # Bot name
        r, g, b = hex_to_rgb(BLUE)
        self.pdf.set_text_color(r, g, b)
        self.pdf.set_font("Helvetica", "", 16)
        self.pdf.cell(self.content_w, 10, f"{BOT_NAME} {BOT_VERSION}",
                     align="C", new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(3)

        # Gold line
        r, g, b = hex_to_rgb(GOLD)
        self.pdf.set_draw_color(r, g, b)
        self.pdf.set_line_width(1)
        self.pdf.line(self.margin + 40, self.pdf.get_y(),
                     self.page_w - self.margin - 40, self.pdf.get_y())
        self.pdf.ln(10)

        # Info block
        self.pdf.set_text_color(80, 80, 80)
        self.pdf.set_font("Helvetica", "", 11)
        info_lines = [
            f"Account: Exness Demo (XAUUSDm)",
            f"Starting Balance: ${balance:.2f}",
            f"Total Trades: {n_trades}",
            f"Report Period: Last {days} days",
            f"Generated: {datetime.now(tz=None).strftime('%Y-%m-%d %H:%M UTC')}",
        ]
        for line in info_lines:
            self.pdf.cell(self.content_w, 7, line, align="C",
                         new_x="LMARGIN", new_y="NEXT")

        self.pdf.ln(30)
        self.pdf.set_text_color(150, 150, 150)
        self.pdf.set_font("Helvetica", "I", 8)
        self.pdf.cell(self.content_w, 5,
                     "PT Surya Inovasi Prioritas (SURIOTA) - Confidential",
                     align="C", new_x="LMARGIN", new_y="NEXT")

    def _page_summary(self, analysis: Dict, chart_paths: Dict[str, str]):
        """Summary page with KPI grid + charts."""
        self.pdf.add_page()
        self._set_navy_header("PERFORMANCE SUMMARY")

        overall = analysis.get("overall", {})

        # KPI grid — 4x2
        self.pdf.set_font("Helvetica", "", 9)
        kpis = [
            ("Total Trades", str(overall.get("total_trades", 0))),
            ("Net P&L", f"${overall.get('net_profit', 0):.2f}"),
            ("Win Rate", f"{overall.get('win_rate', 0):.1%}"),
            ("Profit Factor", safe_pf(overall.get("profit_factor", 0))),
            ("Avg Win", f"${overall.get('avg_win', 0):.2f}"),
            ("Avg Loss", f"${abs(overall.get('avg_loss', 0)):.2f}"),
            ("Max Consec Losses", str(overall.get("max_consecutive_losses", 0))),
            ("Avg Duration", format_duration(overall.get("avg_duration_minutes"))),
        ]

        cell_w = self.content_w / 2
        for i in range(0, len(kpis), 2):
            for j in range(2):
                if i + j < len(kpis):
                    label, value = kpis[i + j]
                    self.pdf.set_font("Helvetica", "B", 8)
                    self.pdf.set_text_color(*hex_to_rgb(NAVY))
                    self.pdf.cell(cell_w * 0.5, 6, label + ":", align="R")
                    self.pdf.set_font("Helvetica", "", 9)
                    self.pdf.set_text_color(*hex_to_rgb(BLUE))
                    self.pdf.cell(cell_w * 0.5, 6, value, align="L")
            self.pdf.ln()

        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(3)

        # Charts
        eq_path = chart_paths.get("equity_curve")
        if eq_path and os.path.exists(eq_path):
            self.pdf.image(eq_path, x=self.margin, w=self.content_w)
            self.pdf.ln(2)

        dd_path = chart_paths.get("drawdown")
        if dd_path and os.path.exists(dd_path):
            self.pdf.image(dd_path, x=self.margin, w=self.content_w)

    @staticmethod
    def _abbrev_session(s: str) -> str:
        """Shorten session names for PDF table."""
        return (s.replace("London-NY Overlap", "LDN/NY")
                 .replace("London Session", "London")
                 .replace("NY Session", "New York")
                 .replace("Asian Session", "Asian"))

    @staticmethod
    def _abbrev_exit(s: str) -> str:
        """Shorten exit reasons for PDF table."""
        import re
        s = (s.replace("Trailing SL Hit (MT5 external)", "Trail SL")
              .replace("SL Hit (MT5 external)", "SL Hit")
              .replace("TP Hit (MT5 external)", "TP Hit")
              .replace("Opposite structure break", "Opp. Break")
              .replace("Session End Profit", "Session Lock")
              .replace(" (MT5 external)", ""))
        # Strip parenthetical details e.g. "(49% of TP, 30min to close)"
        s = re.sub(r"\s*\(.*?\)", "", s)
        # Hard cap at 20 chars
        if len(s) > 20:
            s = s[:18] + ".."
        return s

    @staticmethod
    def _fmt_mfe(val) -> str:
        """Format MFE: must be >= 0, show '-' if zero/missing."""
        if not val:
            return "-"
        v = max(float(val), 0)  # MFE can't be negative
        return f"${v:.2f}" if v > 0 else "-"

    @staticmethod
    def _fmt_mae(val) -> str:
        """Format MAE: must be <= 0, show '-' if zero/missing."""
        if not val:
            return "-"
        v = min(float(val), 0)  # MAE can't be positive
        return f"${v:.2f}" if v < 0 else "-"

    def _page_trade_log(self, trades: List[Dict]):
        """Trade log pages — landscape orientation for wider table."""
        if not trades:
            return

        # Switch to landscape for trade log
        self.pdf.add_page(orientation="L")
        land_w = 297 - 2 * self.margin  # A4 landscape content width

        self._set_navy_header("TRADE LOG")

        col_w = [7, 14, 7, 24, 16, 16, 14, 10, 28, 28, 24, 28, 14, 14, 14, 9]
        headers = ["#", "Ticket", "Dir", "Entry Date/Time", "Entry",
                    "Exit", "P&L", "R:R", "SMC", "Session", "Regime", "Exit Reason",
                    "MFE", "MAE", "Duration", "W/L"]

        total_w = sum(col_w)
        if total_w > land_w:
            scale = land_w / total_w
            col_w = [w * scale for w in col_w]

        self._table_header(col_w, headers)

        for idx, t in enumerate(trades):
            if self.pdf.get_y() > 185:  # Landscape page break earlier
                self.pdf.add_page(orientation="L")
                self._set_navy_header("TRADE LOG (continued)")
                self._table_header(col_w, headers)

            pnl = t.get("profit_usd", 0)
            exit_p = t.get("exit_price", 0)
            wl = "W" if t.get("is_win") else ("L" if t.get("is_loss") else "BE")
            values = [
                str(t.get("trade_no", idx + 1)),
                str(t.get("ticket", ""))[-6:],
                t.get("direction", "")[:1],
                f"{t.get('entry_date', '')} {t.get('entry_time_utc', '')[:5]}",
                f"{t.get('entry_price', 0):.2f}",
                f"{exit_p:.2f}" if exit_p else "-",
                f"${pnl:.2f}",
                f"{t.get('rr_ratio', 0):.2f}",
                t.get("smc_combo", ""),
                self._abbrev_session(t.get("session", "")),
                t.get("regime", "") or "-",
                self._abbrev_exit(t.get("exit_reason", "")),
                self._fmt_mfe(t.get('mfe_usd', 0)),
                self._fmt_mae(t.get('mae_usd', 0)),
                t.get("duration_str", ""),
                wl,
            ]
            self._table_row(col_w, values, alt=(idx % 2 == 1), pnl_col=6)

    def _page_session(self, session_data: Dict, chart_paths: Dict[str, str]):
        """Session analysis page."""
        self.pdf.add_page()
        self._set_navy_header("SESSION ANALYSIS")

        if session_data:
            col_w = [30, 14, 14, 14, 18, 14, 22, 22, 22]
            headers = ["Session", "Trades", "Wins", "Losses", "Win Rate",
                        "PF", "Net P&L", "Avg Win", "Avg Loss"]

            total_w = sum(col_w)
            if total_w > self.content_w:
                scale = self.content_w / total_w
                col_w = [w * scale for w in col_w]

            self._table_header(col_w, headers)

            for idx, (name, s) in enumerate(session_data.items()):
                pf = s["profit_factor"]
                values = [
                    name,
                    str(s["total"]),
                    str(s["wins"]),
                    str(s["losses"]),
                    f"{s['win_rate']:.1%}",
                    safe_pf(pf),
                    f"${s['net_profit']:.2f}",
                    f"${s['avg_win']:.2f}",
                    f"${s['avg_loss']:.2f}",
                ]
                self._table_row(col_w, values, alt=(idx % 2 == 1), pnl_col=6)

        self.pdf.ln(5)
        pie_path = chart_paths.get("session_pie")
        if pie_path and os.path.exists(pie_path):
            self.pdf.image(pie_path, x=self.margin + 30, w=120)

    def _page_regime(self, regime_data: Dict, chart_paths: Dict[str, str]):
        """Regime analysis page."""
        self.pdf.add_page()
        self._set_navy_header("REGIME ANALYSIS")

        if regime_data:
            col_w = [30, 16, 16, 16, 20, 16, 26]
            headers = ["Regime", "Trades", "Wins", "Losses", "Win Rate", "PF", "Net P&L"]

            total_w = sum(col_w)
            if total_w > self.content_w:
                scale = self.content_w / total_w
                col_w = [w * scale for w in col_w]

            self._table_header(col_w, headers)

            for idx, (name, s) in enumerate(regime_data.items()):
                pf = s["profit_factor"]
                values = [
                    name,
                    str(s["total"]),
                    str(s["wins"]),
                    str(s["losses"]),
                    f"{s['win_rate']:.1%}",
                    safe_pf(pf),
                    f"${s['net_profit']:.2f}",
                ]
                self._table_row(col_w, values, alt=(idx % 2 == 1), pnl_col=6)

        self.pdf.ln(5)
        pie_path = chart_paths.get("regime_pie")
        if pie_path and os.path.exists(pie_path):
            self.pdf.image(pie_path, x=self.margin + 30, w=120)

    def _page_smc(self, smc_data: Dict):
        """SMC combo performance page."""
        self.pdf.add_page()
        self._set_navy_header("SMC SIGNAL COMBO PERFORMANCE")

        if smc_data:
            col_w = [40, 16, 16, 20, 26, 26]
            headers = ["SMC Combo", "Trades", "Wins", "Win Rate", "Net P&L", "Avg P&L"]

            total_w = sum(col_w)
            if total_w > self.content_w:
                scale = self.content_w / total_w
                col_w = [w * scale for w in col_w]

            self._table_header(col_w, headers)

            for idx, (combo, s) in enumerate(smc_data.items()):
                values = [
                    combo,
                    str(s["total"]),
                    str(s["wins"]),
                    f"{s['win_rate']:.1%}",
                    f"${s['net_profit']:.2f}",
                    f"${s['avg_profit']:.2f}",
                ]
                self._table_row(col_w, values, alt=(idx % 2 == 1), pnl_col=4)

    def _page_hourly(self, chart_paths: Dict[str, str]):
        """Hourly performance page."""
        self.pdf.add_page()
        self._set_navy_header("HOURLY PERFORMANCE (UTC)")

        hourly_path = chart_paths.get("hourly_perf")
        if hourly_path and os.path.exists(hourly_path):
            self.pdf.image(hourly_path, x=self.margin, w=self.content_w)

        self.pdf.ln(5)
        rr_path = chart_paths.get("rr_dist")
        if rr_path and os.path.exists(rr_path):
            self.pdf.image(rr_path, x=self.margin, w=self.content_w)

    def _page_system_config(self):
        """System configuration page."""
        self.pdf.add_page()
        self._set_navy_header("SYSTEM CONFIGURATION")

        self.pdf.set_font("Helvetica", "", 9)

        config_items = [
            ("Bot", f"{BOT_NAME} {BOT_VERSION}"),
            ("Broker", "Exness Demo"),
            ("Symbol", "XAUUSDm (XAUUSD 3-digit)"),
            ("Timeframe", "M15 primary, H1 context"),
            ("Leverage", "1:100"),
            ("Position Sizing", "Fixed 0.01 lot"),
            ("Max Positions", "1"),
            ("SL Method", "ATR-based (regime-adaptive)"),
            ("TP Method", "ATR-based (6.02x, Optuna-optimized)"),
            ("Exit Stages", "BE@0.77R, Partial@dynamic, Trail@2.72R"),
            ("SMC Version", "V4 (smartmoneyconcepts library)"),
            ("Signals", "BOS + CHoCH + FVG + OB + LiqSweep"),
            ("Scoring", "Adaptive confluence (regime thresholds)"),
            ("Sessions", "Overlap=1.18, London/NY=1.16, Asian=0.75"),
        ]

        # Note: BOT_VERSION constant is authoritative (v4.0.0),
        # settings.yaml may have stale "1.0.0" — do not override.

        col_w = [45, self.content_w - 45]
        for label, value in config_items:
            self.pdf.set_font("Helvetica", "B", 9)
            self.pdf.set_text_color(*hex_to_rgb(NAVY))
            self.pdf.cell(col_w[0], 6, label, border=1)
            self.pdf.set_font("Helvetica", "", 9)
            self.pdf.set_text_color(0, 0, 0)
            self.pdf.cell(col_w[1], 6, value, border=1)
            self.pdf.ln()

        self.pdf.ln(10)
        self.pdf.set_font("Helvetica", "I", 7)
        self.pdf.set_text_color(150, 150, 150)
        self.pdf.cell(self.content_w, 5,
                     f"Report generated by {BOT_NAME} Report Generator",
                     align="C", new_x="LMARGIN", new_y="NEXT")
        self.pdf.cell(self.content_w, 5,
                     "PT Surya Inovasi Prioritas (SURIOTA) - Confidential",
                     align="C")


# ═══════════════════════════════════════════════════════════════════════════
# TradeReportGenerator — Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class TradeReportGenerator:
    """Main orchestrator: load data, enrich, generate all report formats."""

    def __init__(self, days: int = 90, output_dir: str = "reports",
                 formats: str = "all", initial_balance: float = 100.0):
        self.days = days
        self.output_dir = Path(output_dir)
        self.formats = formats
        self.initial_balance = initial_balance

        self.trade_history_dir = project_root / "data" / "trade_history"
        self.journal_dir = project_root / "logs" / "trade_journal"

        self.analyzer = TradeAnalyzer(str(self.trade_history_dir))
        self.trades_enriched: List[Dict] = []
        self.analysis: Dict = {}

    def generate(self):
        """Main entry point — generate all requested reports."""
        print(f"\n{'='*60}")
        print(f"  Trade Journal Report Generator")
        print(f"  {BOT_NAME} {BOT_VERSION}")
        print(f"{'='*60}\n")

        # 1. Load and analyze
        print("[1/6] Loading trade data...")
        n_loaded = self.analyzer.load_trades(days_back=self.days)
        if n_loaded == 0:
            print("  No trade data found. Generating empty reports.")

        print("[2/6] Computing analysis...")
        # Don't call get_full_analysis() — it reloads with default 90 days.
        # Instead, compute stats from the already-loaded trades.
        self.analysis = {
            "loaded": n_loaded,
            "overall": self.analyzer.compute_overall_stats(),
            "by_session": self.analyzer.compute_session_stats(),
            "by_smc_combo": self.analyzer.compute_smc_combo_stats(),
            "by_regime": self.analyzer.compute_regime_stats(),
            "by_hour": self.analyzer.compute_time_stats(),
        }

        # 3. Build enriched trade list
        print("[3/6] Enriching trades from journals...")
        self.trades_enriched = self._build_enriched_trades()
        print(f"  Found {len(self.trades_enriched)} closed trades")

        # Compute aggregations
        daily_agg = self._aggregate_by_period("day")
        weekly_agg = self._aggregate_by_period("week")
        monthly_agg = self._aggregate_by_period("month")

        # Ensure output dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(tz=None).strftime("%Y-%m-%d")

        do_csv = self.formats in ("all", "csv")
        do_xlsx = self.formats in ("all", "xlsx")
        do_pdf = self.formats in ("all", "pdf")

        # 4. Generate charts (needed for PDF and XLSX reference)
        chart_paths = {}
        chart_dir = None
        if (do_pdf or do_xlsx) and HAS_MATPLOTLIB:
            print("[4/6] Generating charts...")
            chart_dir = tempfile.mkdtemp(prefix="trade_report_charts_")
            chart_paths = self._generate_charts(chart_dir)
        else:
            print("[4/6] Charts skipped (matplotlib not available or not needed)")

        # 5. CSV
        if do_csv:
            print("[5/6] Generating CSV...")
            csv_path = self.output_dir / f"trade_report_{date_str}.csv"
            self._generate_csv(csv_path)
            print(f"  -> {csv_path}")
        else:
            print("[5/6] CSV skipped")

        # 6. XLSX
        if do_xlsx:
            if HAS_XLSXWRITER:
                print("[6/6] Generating Excel report...")
                xlsx_path = self.output_dir / f"trade_report_{date_str}.xlsx"
                builder = ExcelReportBuilder(str(xlsx_path))
                builder.build(
                    self.trades_enriched, self.analysis, self.initial_balance,
                    daily_agg, weekly_agg, monthly_agg,
                )
                print(f"  -> {xlsx_path}")
            else:
                print("[6/6] XLSX skipped (xlsxwriter not installed)")
        else:
            print("[6/6] XLSX skipped")

        # 7. PDF
        if do_pdf:
            if HAS_FPDF:
                print("[+] Generating PDF report...")
                pdf_path = self.output_dir / f"trade_report_{date_str}.pdf"
                builder = PDFReportBuilder(str(pdf_path))
                builder.build(
                    self.trades_enriched, self.analysis, self.initial_balance,
                    daily_agg, chart_paths,
                    self.analysis.get("by_session", {}),
                    self.analysis.get("by_regime", {}),
                    self.analysis.get("by_smc_combo", {}),
                    self.analysis.get("by_hour", {}),
                    days=self.days,
                )
                print(f"  -> {pdf_path}")
            else:
                print("[+] PDF skipped (fpdf2 not installed)")

        # Cleanup temp charts
        if chart_dir and os.path.exists(chart_dir):
            shutil.rmtree(chart_dir, ignore_errors=True)

        print(f"\nDone! Reports saved to: {self.output_dir.resolve()}")

    def _build_enriched_trades(self) -> List[Dict]:
        """Build enriched trade DataFrame from closed trades + journal data."""
        closed = self.analyzer.get_closed_trades()
        if not closed:
            return []

        # Build open_map for SL/TP/direction from CSV OPEN rows
        open_map = self.analyzer._build_open_map()

        enriched = []
        cumulative_pnl = 0.0
        peak_equity = self.initial_balance

        for idx, trade in enumerate(closed):
            ticket = trade.get("ticket", "")
            open_row = open_map.get(ticket, {})

            # Journal enrichment
            journal = self._load_journal(ticket)
            entry_data = (journal.get("entry") or {}) if journal else {}
            exit_data = (journal.get("exit") or {}) if journal else {}

            # Direction
            direction = (open_row.get("direction") or
                        (journal.get("direction") if journal else "") or "")

            # Dates/times
            entry_date = open_row.get("date", "")
            entry_time_utc = open_row.get("time_utc", "")
            entry_time_wib = open_row.get("time_wib", "")
            exit_date = trade.get("date", "")
            exit_time_utc = trade.get("time_utc", "")
            exit_time_wib = trade.get("time_wib", "")

            # Prices — fallback to journal if CSV exit price is 0
            entry_price = trade.get("entry_price", 0)
            exit_price = trade.get("exit_price", 0)
            if not exit_price and exit_data.get("price"):
                exit_price = exit_data["price"]
            # Compute exit price from P&L if still 0 (MT5 external close)
            profit_usd_raw = trade.get("profit_num", 0)
            if not exit_price and entry_price and profit_usd_raw:
                try:
                    vol = float(open_row.get("volume") or
                               (journal.get("volume") if journal else 0) or 0.01)
                    pip_value = vol * 100  # XAUUSD: 1 pip = $1 per 0.01 lot
                    if pip_value > 0:
                        pips_moved = profit_usd_raw / pip_value
                        if direction == "BUY":
                            exit_price = entry_price + pips_moved
                        elif direction == "SELL":
                            exit_price = entry_price - pips_moved
                except (ValueError, TypeError, ZeroDivisionError):
                    pass

            # SL/TP from OPEN row or journal
            try:
                sl = float(open_row.get("sl") or entry_data.get("sl") or 0)
            except (ValueError, TypeError):
                sl = 0.0
            try:
                tp = float(open_row.get("tp") or entry_data.get("tp") or 0)
            except (ValueError, TypeError):
                tp = 0.0

            # SL/TP in pips
            sl_pips = entry_data.get("sl_pips", 0)
            tp_pips = entry_data.get("tp_pips", 0)
            if not sl_pips and sl and entry_price:
                sl_pips = abs(entry_price - sl)
            if not tp_pips and tp and entry_price:
                tp_pips = abs(tp - entry_price)

            # Volume
            try:
                volume = float(open_row.get("volume") or
                              (journal.get("volume") if journal else 0) or 0.01)
            except (ValueError, TypeError):
                volume = 0.01

            # P&L
            profit_usd = trade.get("profit_num", 0)

            # R:R from journal exit or compute
            rr_ratio = exit_data.get("rr_final", 0)
            if not rr_ratio and sl_pips and profit_usd:
                # Approximate: rr = profit / (sl_pips * vol * contract_factor)
                sl_dollar = sl_pips * volume * 100  # rough XAUUSD
                if sl_dollar > 0:
                    rr_ratio = profit_usd / sl_dollar

            # Duration
            duration_min = trade.get("duration_minutes") or exit_data.get("duration_minutes")
            duration_str = format_duration(duration_min)

            # Journal fields
            mfe_usd = exit_data.get("mfe_usd", 0)
            mae_usd = exit_data.get("mae_usd", 0)
            stage_reached = exit_data.get("stage_reached", "")
            snapshots_count = exit_data.get("snapshots_count", 0)

            # Exit reason
            exit_reason = trade.get("comment", "") or exit_data.get("exit_reason", "")

            # Cumulative fields
            cumulative_pnl += profit_usd
            equity = self.initial_balance + cumulative_pnl
            if equity > peak_equity:
                peak_equity = equity
            drawdown_pct = ((peak_equity - equity) / peak_equity * 100
                           if peak_equity > 0 else 0)

            # Entry hour for hourly analysis
            entry_hour = ""
            if entry_time_utc:
                try:
                    entry_hour = int(entry_time_utc.split(":")[0])
                except (ValueError, IndexError):
                    entry_hour = ""

            # Week/month labels
            week_label = ""
            month_label = ""
            if entry_date:
                try:
                    dt = datetime.strptime(entry_date, "%Y-%m-%d")
                    iso_year, iso_week, _ = dt.isocalendar()
                    week_label = f"{iso_year}-W{iso_week:02d}"
                    month_label = dt.strftime("%Y-%m")
                except ValueError:
                    pass

            enriched.append({
                "trade_no": idx + 1,
                "ticket": ticket,
                "direction": direction,
                "entry_date": entry_date,
                "entry_time_utc": entry_time_utc,
                "entry_time_wib": entry_time_wib,
                "exit_date": exit_date,
                "exit_time_utc": exit_time_utc,
                "exit_time_wib": exit_time_wib,
                "entry_price": round(entry_price, 3),
                "exit_price": round(exit_price, 3),
                "sl": round(sl, 3),
                "tp": round(tp, 3),
                "sl_pips": round(sl_pips, 2),
                "tp_pips": round(tp_pips, 2),
                "volume": volume,
                "profit_usd": round(profit_usd, 2),
                "rr_ratio": round(rr_ratio, 3) if rr_ratio else 0,
                "duration_min": duration_min,
                "duration_str": duration_str,
                "confluence": round(trade.get("confluence", 0), 4),
                "smc_combo": trade.get("smc_combo", ""),
                "session": trade.get("session", ""),
                "regime": trade.get("regime", ""),
                "entry_reason": entry_data.get("smc_signals", ""),
                "exit_reason": exit_reason,
                "mfe_usd": round(mfe_usd, 2),
                "mae_usd": round(mae_usd, 2),
                "stage_reached": stage_reached,
                "snapshots_count": snapshots_count,
                "is_win": trade.get("is_win", False),
                "is_loss": trade.get("is_loss", False),
                "is_breakeven": trade.get("is_breakeven", False),
                "cumulative_pnl": round(cumulative_pnl, 2),
                "equity": round(equity, 2),
                "drawdown_pct": round(drawdown_pct, 2),
                "peak_equity": round(peak_equity, 2),
                "week_label": week_label,
                "month_label": month_label,
                "entry_hour": entry_hour,
            })

        return enriched

    def _load_journal(self, ticket: str) -> Optional[Dict]:
        """Load journal JSON for a ticket."""
        if not self.journal_dir.exists():
            return None

        # Try finding the file by ticket suffix
        for jf in self.journal_dir.glob(f"*_{ticket}.json"):
            try:
                with open(jf) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _aggregate_by_period(self, period: str) -> List[Dict]:
        """Aggregate trades by day/week/month."""
        if not self.trades_enriched:
            return []

        groups = defaultdict(list)
        for t in self.trades_enriched:
            if period == "day":
                key = t.get("entry_date", "Unknown")
            elif period == "week":
                key = t.get("week_label", "Unknown")
            elif period == "month":
                key = t.get("month_label", "Unknown")
            else:
                key = "Unknown"
            groups[key].append(t)

        result = []
        cumulative = 0.0
        for key in sorted(groups.keys()):
            trades = groups[key]
            wins = [t for t in trades if t["is_win"]]
            losses = [t for t in trades if t["is_loss"]]
            net = sum(t["profit_usd"] for t in trades)
            cumulative += net

            gross_profit = sum(t["profit_usd"] for t in wins)
            gross_loss = abs(sum(t["profit_usd"] for t in losses))
            pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            result.append({
                "period": key,
                "trades": len(trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(trades) if trades else 0,
                "pf": pf,
                "net_pnl": net,
                "cumulative": cumulative,
            })

        return result

    def _generate_charts(self, chart_dir: str) -> Dict[str, str]:
        """Generate all chart PNGs."""
        cg = ChartGenerator(chart_dir)
        paths = {}

        if not self.trades_enriched:
            return paths

        # Equity curve
        equities = [t["equity"] for t in self.trades_enriched]
        dates = [t["entry_date"] for t in self.trades_enriched]
        paths["equity_curve"] = cg.equity_curve(equities, dates, self.initial_balance)

        # Drawdown
        drawdowns = [t["drawdown_pct"] for t in self.trades_enriched]
        paths["drawdown"] = cg.drawdown_curve(drawdowns)

        # Daily P&L
        daily_pnl = defaultdict(float)
        for t in self.trades_enriched:
            daily_pnl[t.get("entry_date", "Unknown")] += t["profit_usd"]
        paths["daily_pnl"] = cg.daily_pnl_bars(dict(sorted(daily_pnl.items())))

        # Win/loss pie
        overall = self.analysis.get("overall", {})
        paths["win_loss_pie"] = cg.win_loss_pie(
            overall.get("wins", 0),
            overall.get("losses", 0),
            overall.get("breakevens", 0),
        )

        # Session pie
        paths["session_pie"] = cg.session_pie(
            self.analysis.get("by_session", {}))

        # Regime pie
        paths["regime_pie"] = cg.regime_pie(
            self.analysis.get("by_regime", {}))

        # Hourly performance
        paths["hourly_perf"] = cg.hourly_performance(
            self.analysis.get("by_hour", {}))

        # R:R distribution
        rr_values = [t["rr_ratio"] for t in self.trades_enriched
                     if t["rr_ratio"] and t["rr_ratio"] != 0]
        paths["rr_dist"] = cg.rr_distribution(rr_values)

        return paths

    def _generate_csv(self, filepath: Path):
        """Generate CSV report."""
        columns = [
            "trade_no", "ticket", "direction", "entry_date", "entry_time_utc",
            "entry_time_wib", "exit_date", "exit_time_utc", "exit_time_wib",
            "entry_price", "exit_price", "sl", "tp", "sl_pips", "tp_pips",
            "volume", "profit_usd", "rr_ratio", "duration_min", "duration_str",
            "confluence", "smc_combo", "session", "regime",
            "entry_reason", "exit_reason",
            "mfe_usd", "mae_usd", "stage_reached", "snapshots_count",
            "is_win", "is_loss", "is_breakeven",
            "cumulative_pnl", "equity", "drawdown_pct",
            "week_label", "month_label", "entry_hour",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for t in self.trades_enriched:
                writer.writerow(t)


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate professional trade journal reports (CSV, XLSX, PDF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_trade_report.py
  python scripts/generate_trade_report.py --days 30 --balance 100
  python scripts/generate_trade_report.py --format xlsx --output reports/
  python scripts/generate_trade_report.py --format pdf --days 7
        """,
    )
    parser.add_argument("--days", type=int, default=90,
                        help="Number of days of trade history to include (default: 90)")
    parser.add_argument("--output", type=str, default="reports",
                        help="Output directory for reports (default: reports/)")
    parser.add_argument("--format", type=str, default="all",
                        choices=["all", "csv", "xlsx", "pdf"],
                        help="Report format to generate (default: all)")
    parser.add_argument("--balance", type=float, default=100.0,
                        help="Starting balance for equity curve (default: 100.0)")

    args = parser.parse_args()

    # Check optional dependencies
    missing = []
    if args.format in ("all", "xlsx") and not HAS_XLSXWRITER:
        missing.append("xlsxwriter (pip install xlsxwriter)")
    if args.format in ("all", "pdf") and not HAS_FPDF:
        missing.append("fpdf2 (pip install fpdf2)")
    if args.format in ("all", "pdf", "xlsx") and not HAS_MATPLOTLIB:
        missing.append("matplotlib (pip install matplotlib)")

    if missing:
        print("WARNING: Missing optional dependencies:")
        for m in missing:
            print(f"  - {m}")
        print("Some report formats will be skipped.\n")

    generator = TradeReportGenerator(
        days=args.days,
        output_dir=args.output,
        formats=args.format,
        initial_balance=args.balance,
    )
    generator.generate()


if __name__ == "__main__":
    main()
