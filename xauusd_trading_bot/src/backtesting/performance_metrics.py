"""
Performance Metrics
Calculate trading performance metrics for backtesting.
"""

import polars as pl
import numpy as np
from typing import Dict, List
from datetime import datetime
from ..bot_logger import get_logger


class PerformanceMetrics:
    """Calculate trading performance metrics."""

    def __init__(self):
        """Initialize performance metrics calculator."""
        self.logger = get_logger()

    def calculate_all_metrics(
        self,
        trades: List[Dict],
        initial_balance: float,
        final_balance: float,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Calculate all performance metrics.

        Args:
            trades: List of completed trades
            initial_balance: Starting balance
            final_balance: Ending balance
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Dictionary of all metrics
        """
        if not trades:
            return self._empty_metrics()

        # Convert to DataFrame for easier analysis
        df = self._trades_to_dataframe(trades)

        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(df.filter(pl.col("profit") > 0))
        losing_trades = len(df.filter(pl.col("profit") < 0))
        breakeven_trades = len(df.filter(pl.col("profit") == 0))

        # Win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Profit metrics
        total_profit = df["profit"].sum()
        gross_profit = df.filter(pl.col("profit") > 0)["profit"].sum()
        gross_loss = abs(df.filter(pl.col("profit") < 0)["profit"].sum())

        # Profit factor
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

        # Average metrics
        avg_profit = df["profit"].mean()
        avg_win = df.filter(pl.col("profit") > 0)["profit"].mean() if winning_trades > 0 else 0
        avg_loss = df.filter(pl.col("profit") < 0)["profit"].mean() if losing_trades > 0 else 0

        # Risk/reward ratio
        avg_rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Largest win/loss
        largest_win = df["profit"].max()
        largest_loss = df["profit"].min()

        # Consecutive wins/losses
        max_consecutive_wins = self._calculate_max_consecutive(df, "profit", ">", 0)
        max_consecutive_losses = self._calculate_max_consecutive(df, "profit", "<", 0)

        # Drawdown metrics
        equity_curve = self._calculate_equity_curve(df, initial_balance)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        max_drawdown_percent = (max_drawdown / initial_balance * 100) if initial_balance > 0 else 0

        # Return metrics
        total_return = final_balance - initial_balance
        total_return_percent = (total_return / initial_balance * 100) if initial_balance > 0 else 0

        # Time metrics
        duration_days = (end_date - start_date).days
        trades_per_day = total_trades / duration_days if duration_days > 0 else 0

        # Sharpe ratio (simplified - assuming risk-free rate = 0)
        sharpe_ratio = self._calculate_sharpe_ratio(df["profit"].to_numpy())

        # Expectancy
        expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

        return {
            # Trade counts
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "breakeven_trades": breakeven_trades,
            "win_rate": round(win_rate, 2),

            # Profit metrics
            "total_profit": round(total_profit, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 2),

            # Average metrics
            "avg_profit": round(avg_profit, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_rr_ratio": round(avg_rr_ratio, 2),

            # Extremes
            "largest_win": round(largest_win, 2),
            "largest_loss": round(largest_loss, 2),
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,

            # Drawdown
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_percent": round(max_drawdown_percent, 2),

            # Returns
            "initial_balance": round(initial_balance, 2),
            "final_balance": round(final_balance, 2),
            "total_return": round(total_return, 2),
            "total_return_percent": round(total_return_percent, 2),

            # Risk metrics
            "sharpe_ratio": round(sharpe_ratio, 2),
            "expectancy": round(expectancy, 2),

            # Time metrics
            "duration_days": duration_days,
            "trades_per_day": round(trades_per_day, 2),

            # Dates
            "start_date": start_date,
            "end_date": end_date,
        }

    def _trades_to_dataframe(self, trades: List[Dict]) -> pl.DataFrame:
        """Convert trade list to Polars DataFrame."""
        return pl.DataFrame(trades)

    def _empty_metrics(self) -> Dict:
        """Return empty metrics for when there are no trades."""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "win_rate": 0.0,
            "total_profit": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "avg_profit": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_rr_ratio": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "max_drawdown": 0.0,
            "max_drawdown_percent": 0.0,
            "initial_balance": 0.0,
            "final_balance": 0.0,
            "total_return": 0.0,
            "total_return_percent": 0.0,
            "sharpe_ratio": 0.0,
            "expectancy": 0.0,
            "duration_days": 0,
            "trades_per_day": 0.0,
        }

    def _calculate_max_consecutive(
        self, df: pl.DataFrame, column: str, operator: str, value: float
    ) -> int:
        """Calculate maximum consecutive occurrences."""
        if operator == ">":
            mask = df[column] > value
        elif operator == "<":
            mask = df[column] < value
        else:
            mask = df[column] == value

        # Convert to numpy for easier processing
        mask_array = mask.to_numpy()

        max_consecutive = 0
        current_consecutive = 0

        for val in mask_array:
            if val:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_equity_curve(
        self, df: pl.DataFrame, initial_balance: float
    ) -> np.ndarray:
        """Calculate equity curve."""
        profits = df["profit"].to_numpy()
        equity = np.zeros(len(profits) + 1)
        equity[0] = initial_balance

        for i, profit in enumerate(profits):
            equity[i + 1] = equity[i] + profit

        return equity

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) == 0:
            return 0

        peak = equity_curve[0]
        max_dd = 0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _calculate_sharpe_ratio(
        self, returns: np.ndarray, risk_free_rate: float = 0
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Array of trade returns
            risk_free_rate: Risk-free rate (default 0)

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0

        sharpe = (mean_return - risk_free_rate) / std_return
        return sharpe

    def generate_report(self, metrics: Dict) -> str:
        """
        Generate human-readable performance report.

        Args:
            metrics: Performance metrics dictionary

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("BACKTEST PERFORMANCE REPORT")
        report.append("=" * 80)

        # Period
        report.append("\nPERIOD:")
        start = metrics.get('start_date')
        end = metrics.get('end_date')
        report.append(f"  Start Date: {start.strftime('%Y-%m-%d') if hasattr(start, 'strftime') else 'N/A'}")
        report.append(f"  End Date: {end.strftime('%Y-%m-%d') if hasattr(end, 'strftime') else 'N/A'}")
        report.append(f"  Duration: {metrics.get('duration_days', 0)} days")

        # Trade Summary
        report.append("\nTRADE SUMMARY:")
        report.append(f"  Total Trades: {metrics['total_trades']}")
        report.append(f"  Winning Trades: {metrics['winning_trades']}")
        report.append(f"  Losing Trades: {metrics['losing_trades']}")
        report.append(f"  Breakeven Trades: {metrics['breakeven_trades']}")
        report.append(f"  Win Rate: {metrics['win_rate']:.2f}%")
        report.append(f"  Trades Per Day: {metrics['trades_per_day']:.2f}")

        # Profit Metrics
        report.append("\nPROFIT METRICS:")
        report.append(f"  Total Profit: ${metrics['total_profit']:.2f}")
        report.append(f"  Gross Profit: ${metrics['gross_profit']:.2f}")
        report.append(f"  Gross Loss: ${metrics['gross_loss']:.2f}")
        report.append(f"  Profit Factor: {metrics['profit_factor']:.2f}")

        # Average Metrics
        report.append("\nAVERAGE METRICS:")
        report.append(f"  Avg Profit Per Trade: ${metrics['avg_profit']:.2f}")
        report.append(f"  Avg Win: ${metrics['avg_win']:.2f}")
        report.append(f"  Avg Loss: ${metrics['avg_loss']:.2f}")
        report.append(f"  Avg Risk/Reward: {metrics['avg_rr_ratio']:.2f}")

        # Extremes
        report.append("\nEXTREMES:")
        report.append(f"  Largest Win: ${metrics['largest_win']:.2f}")
        report.append(f"  Largest Loss: ${metrics['largest_loss']:.2f}")
        report.append(f"  Max Consecutive Wins: {metrics['max_consecutive_wins']}")
        report.append(f"  Max Consecutive Losses: {metrics['max_consecutive_losses']}")

        # Returns
        report.append("\nRETURNS:")
        report.append(f"  Initial Balance: ${metrics['initial_balance']:.2f}")
        report.append(f"  Final Balance: ${metrics['final_balance']:.2f}")
        report.append(f"  Total Return: ${metrics['total_return']:.2f}")
        report.append(f"  Total Return %: {metrics['total_return_percent']:.2f}%")

        # Risk Metrics
        report.append("\nRISK METRICS:")
        report.append(f"  Max Drawdown: ${metrics['max_drawdown']:.2f}")
        report.append(f"  Max Drawdown %: {metrics['max_drawdown_percent']:.2f}%")
        report.append(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        report.append(f"  Expectancy: ${metrics['expectancy']:.2f}")

        # Assessment
        report.append("\nASSESSMENT:")
        assessment = self._assess_performance(metrics)
        for line in assessment:
            report.append(f"  {line}")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def regime_breakdown_metrics(
        self, trades: List[Dict], regime_labels: List[str]
    ) -> Dict:
        """
        Calculate per-regime performance breakdown.

        Args:
            trades: List of completed trades (must have 'regime' field or use regime_labels)
            regime_labels: Regime label for each trade (aligned by index)

        Returns:
            Dict keyed by regime name with per-regime metrics
        """
        if not trades:
            return {}

        # Group trades by regime
        regime_groups = {}
        for i, trade in enumerate(trades):
            regime = trade.get("regime", regime_labels[i] if i < len(regime_labels) else "UNKNOWN")
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append(trade)

        breakdown = {}
        for regime, group in regime_groups.items():
            n = len(group)
            wins = [t for t in group if t.get("profit", 0) > 0]
            losses = [t for t in group if t.get("profit", 0) < 0]
            total_profit = sum(t.get("profit", 0) for t in group)
            gross_profit = sum(t.get("profit", 0) for t in wins)
            gross_loss = abs(sum(t.get("profit", 0) for t in losses))

            breakdown[regime] = {
                "trades": n,
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(len(wins) / n * 100, 2) if n > 0 else 0,
                "total_profit": round(total_profit, 2),
                "avg_profit": round(total_profit / n, 2) if n > 0 else 0,
                "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
            }

        return breakdown

    def _assess_performance(self, metrics: Dict) -> List[str]:
        """Assess strategy performance."""
        assessment = []

        # Win rate assessment
        if metrics["win_rate"] >= 60:
            assessment.append("✓ Excellent win rate (>= 60%)")
        elif metrics["win_rate"] >= 55:
            assessment.append("✓ Good win rate (>= 55%)")
        elif metrics["win_rate"] >= 50:
            assessment.append("⚠ Acceptable win rate (>= 50%)")
        else:
            assessment.append("✗ Low win rate (< 50%)")

        # Profit factor assessment
        if metrics["profit_factor"] >= 2.0:
            assessment.append("✓ Excellent profit factor (>= 2.0)")
        elif metrics["profit_factor"] >= 1.5:
            assessment.append("✓ Good profit factor (>= 1.5)")
        elif metrics["profit_factor"] >= 1.2:
            assessment.append("⚠ Acceptable profit factor (>= 1.2)")
        else:
            assessment.append("✗ Poor profit factor (< 1.2)")

        # Risk/Reward assessment
        if metrics["avg_rr_ratio"] >= 2.0:
            assessment.append("✓ Excellent R/R ratio (>= 2.0)")
        elif metrics["avg_rr_ratio"] >= 1.5:
            assessment.append("✓ Good R/R ratio (>= 1.5)")
        elif metrics["avg_rr_ratio"] >= 1.0:
            assessment.append("⚠ Acceptable R/R ratio (>= 1.0)")
        else:
            assessment.append("✗ Poor R/R ratio (< 1.0)")

        # Drawdown assessment
        if metrics["max_drawdown_percent"] <= 10:
            assessment.append("✓ Low drawdown (<= 10%)")
        elif metrics["max_drawdown_percent"] <= 20:
            assessment.append("⚠ Moderate drawdown (<= 20%)")
        else:
            assessment.append("✗ High drawdown (> 20%)")

        # Overall assessment
        if metrics["total_return_percent"] > 0:
            assessment.append(f"✓ Profitable strategy (+{metrics['total_return_percent']:.2f}%)")
        else:
            assessment.append(f"✗ Unprofitable strategy ({metrics['total_return_percent']:.2f}%)")

        return assessment
