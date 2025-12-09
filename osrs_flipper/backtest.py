"""Trade simulation and backtesting for OSRS GE flipping.

This module provides tools for simulating trades on historical price data,
calculating returns, and backtesting trading strategies.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .tax import calculate_ge_tax, calculate_net_profit, is_tax_exempt
from .features import calculate_percentile_rank


class TradeSimulator:
    """Simulates trades on historical price data."""

    def __init__(self, prices: pd.DataFrame):
        """
        Initialize with historical prices.

        Args:
            prices: DataFrame with columns: timestamp, mid_price
        """
        self.prices = prices.copy()
        if "timestamp" in self.prices.columns:
            self.prices = self.prices.set_index("timestamp")

    def execute_trade(
        self,
        entry_day: int,
        exit_day: int,
        item_name: str,
    ) -> Dict[str, Any]:
        """
        Execute a simulated trade.

        Args:
            entry_day: Index of entry (buy) day
            exit_day: Index of exit (sell) day
            item_name: Name of item for tax calculation

        Returns:
            Dict with: entry_day, exit_day, hold_days, entry_price, exit_price,
            gross_profit, tax, net_profit, gross_roi, net_roi, is_tax_exempt
        """
        entry_price = int(self.prices.iloc[entry_day]["mid_price"])
        exit_price = int(self.prices.iloc[exit_day]["mid_price"])

        hold_days = exit_day - entry_day
        gross_profit = exit_price - entry_price
        tax = calculate_ge_tax(exit_price, item_name)
        net_profit = calculate_net_profit(entry_price, exit_price, item_name)

        gross_roi = (gross_profit / entry_price) * 100 if entry_price > 0 else 0
        net_roi = (net_profit / entry_price) * 100 if entry_price > 0 else 0

        return {
            "entry_day": entry_day,
            "exit_day": exit_day,
            "hold_days": hold_days,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_profit": gross_profit,
            "tax": tax,
            "net_profit": net_profit,
            "gross_roi": round(gross_roi, 2),
            "net_roi": round(net_roi, 2),
            "is_tax_exempt": is_tax_exempt(item_name),
        }


class SignalBacktester:
    """Backtest trading signals on historical data."""

    def __init__(self, df: pd.DataFrame, item_name: str):
        """
        Initialize backtester with historical data.

        Args:
            df: DataFrame with columns: timestamp, mid_price
            item_name: Name of item for tax calculation
        """
        self.df = df.copy()
        self.item_name = item_name
        self.sim = TradeSimulator(df)

    def backtest_oversold_signal(
        self,
        percentile_threshold: int = 20,
        hold_days: int = 30,
        lookback: int = 90,
    ) -> Dict[str, Any]:
        """
        Backtest oversold signal strategy.

        Entry: When percentile rank <= threshold
        Exit: After hold_days

        Args:
            percentile_threshold: Enter when percentile <= this (default 20)
            hold_days: Hold for this many days (default 30)
            lookback: Lookback window for percentile calculation (default 90)

        Returns:
            Dict with: trades, num_trades, win_rate, avg_return, total_profit,
            max_drawdown, sharpe
        """
        prices = self.df["mid_price"]
        percentiles = calculate_percentile_rank(prices, window=lookback)

        trades = []
        i = lookback  # Start after enough data for percentile

        while i < len(prices) - hold_days:
            if percentiles.iloc[i] <= percentile_threshold:
                # Entry signal
                entry_day = i
                exit_day = min(i + hold_days, len(prices) - 1)

                trade = self.sim.execute_trade(entry_day, exit_day, self.item_name)
                trades.append(trade)

                # Skip to after exit
                i = exit_day + 1
            else:
                i += 1

        if not trades:
            return {
                "trades": [],
                "num_trades": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "total_profit": 0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
            }

        # Calculate statistics
        profits = np.array([t["net_profit"] for t in trades])
        rois = np.array([t["net_roi"] for t in trades])

        wins = int(np.sum(profits > 0))

        # Calculate max drawdown
        cumulative = np.cumsum(profits)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / np.maximum(peak, 1)
        max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        # Calculate Sharpe ratio (annualized)
        avg_roi = np.mean(rois)
        std_roi = np.std(rois) if len(rois) > 1 else 1.0
        sharpe = (avg_roi / std_roi) * np.sqrt(12) if std_roi > 0 else 0.0  # Monthly -> Annual

        return {
            "trades": trades,
            "num_trades": len(trades),
            "win_rate": round(wins / len(trades), 3),
            "avg_return": round(float(np.mean(rois)), 2),
            "total_profit": int(sum(profits)),
            "max_drawdown": round(max_drawdown, 3),
            "sharpe": round(float(sharpe), 2),
        }

    def walk_forward_test(
        self,
        train_days: int = 90,
        test_days: int = 30,
        hold_days: int = 14,
        percentile_threshold: int = 20,
    ) -> Dict[str, Any]:
        """
        Walk-forward validation: train on past, test on future.

        Args:
            train_days: Days in training window
            test_days: Days in test window
            hold_days: Hold period for trades
            percentile_threshold: Entry threshold

        Returns:
            Dict with: periods, overall_win_rate, overall_avg_return, overall_trades
        """
        periods = []
        total_len = len(self.df)

        start = 0
        while start + train_days + test_days <= total_len:
            # Test window
            test_start = start + train_days
            test_end = test_start + test_days

            # Create test subset
            test_df = self.df.iloc[test_start:test_end].reset_index(drop=True)

            if len(test_df) < hold_days + 10:
                start += test_days
                continue

            # Run backtest on test window
            test_backtester = SignalBacktester(test_df, self.item_name)
            result = test_backtester.backtest_oversold_signal(
                percentile_threshold=percentile_threshold,
                hold_days=hold_days,
                lookback=min(30, len(test_df) - hold_days - 1),
            )

            periods.append({
                "period_start": test_start,
                "period_end": test_end,
                "num_trades": result["num_trades"],
                "win_rate": result["win_rate"],
                "avg_return": result["avg_return"],
                "total_profit": result["total_profit"],
            })

            # Move to next period
            start += test_days

        if not periods:
            return {
                "periods": [],
                "overall_win_rate": 0.0,
                "overall_avg_return": 0.0,
                "overall_trades": 0,
            }

        # Aggregate results
        num_trades_arr = np.array([p["num_trades"] for p in periods])
        win_rates_arr = np.array([p["win_rate"] for p in periods])
        avg_returns_arr = np.array([p["avg_return"] for p in periods])

        total_trades = int(num_trades_arr.sum())
        total_wins = float((win_rates_arr * num_trades_arr).sum())
        total_return = float((avg_returns_arr * num_trades_arr).sum())

        return {
            "periods": periods,
            "overall_win_rate": round(total_wins / total_trades, 3) if total_trades > 0 else 0.0,
            "overall_avg_return": round(total_return / total_trades, 2) if total_trades > 0 else 0.0,
            "overall_trades": total_trades,
        }
