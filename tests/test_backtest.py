import pytest
import pandas as pd
from osrs_flipper.backtest import TradeSimulator


class TestTradeSimulator:
    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        return pd.DataFrame({
            "timestamp": list(range(30)),
            "mid_price": [1000, 1050, 1100, 1080, 1120, 1150, 1200, 1180, 1220, 1250] * 3
        })

    def test_buy_and_sell_profit(self, sample_prices):
        """Test profitable trade with tax."""
        sim = TradeSimulator(sample_prices)
        result = sim.execute_trade(0, 5, "Abyssal whip")  # buy 1000, sell 1150

        assert result["entry_price"] == 1000
        assert result["exit_price"] == 1150
        assert result["hold_days"] == 5
        assert result["gross_profit"] == 150
        assert result["tax"] == 23  # 2% of 1150 = 23
        assert result["net_profit"] == 150 - 23  # 127
        assert result["is_tax_exempt"] == False

    def test_loss_trade(self, sample_prices):
        """Test losing trade with tax."""
        # Create loss scenario
        loss_prices = pd.DataFrame({
            "timestamp": [0, 1, 2],
            "mid_price": [1000, 800, 900]
        })
        sim = TradeSimulator(loss_prices)
        result = sim.execute_trade(0, 1, "Dragon bones")  # buy 1000, sell 800

        assert result["entry_price"] == 1000
        assert result["exit_price"] == 800
        assert result["gross_profit"] == -200
        assert result["tax"] == 16  # 2% of 800
        assert result["net_profit"] == 800 - 16 - 1000  # -216
        assert result["net_roi"] < result["gross_roi"]  # Tax makes loss worse

    def test_tax_exempt_item(self, sample_prices):
        """Test trade with tax-exempt item."""
        sim = TradeSimulator(sample_prices)
        result = sim.execute_trade(0, 5, "Bronze arrow")  # Tax exempt

        assert result["tax"] == 0
        assert result["is_tax_exempt"] == True
        assert result["gross_profit"] == result["net_profit"]

    def test_roi_calculation(self, sample_prices):
        """Test ROI percentage calculation."""
        sim = TradeSimulator(sample_prices)
        result = sim.execute_trade(0, 5, "Abyssal whip")

        # gross_roi = (150 / 1000) * 100 = 15%
        assert result["gross_roi"] == 15.0
        # net_roi = (127 / 1000) * 100 = 12.7%
        assert result["net_roi"] == 12.7


class TestSignalBacktester:
    @pytest.fixture
    def crash_recovery_prices(self):
        """Price data with crash-then-recovery pattern."""
        # Start at 1000, crash to 700, recover to 1100
        prices = (
            [1000] * 30 +  # Stable
            [1000 - i * 10 for i in range(30)] +  # Crash to 700
            [700 + i * 13 for i in range(30)] +  # Recovery to 1090
            [1100] * 10  # Stable at top
        )
        return pd.DataFrame({
            "timestamp": list(range(len(prices))),
            "mid_price": prices,
        })

    def test_backtest_oversold_signal(self, crash_recovery_prices):
        """Oversold signal should catch the crash-recovery."""
        from osrs_flipper.backtest import SignalBacktester

        bt = SignalBacktester(crash_recovery_prices, "Abyssal whip")
        result = bt.backtest_oversold_signal(
            percentile_threshold=20,
            hold_days=20,
            lookback=30,
        )

        assert result["num_trades"] >= 1
        assert "win_rate" in result
        assert "sharpe" in result
        assert "max_drawdown" in result

    def test_walk_forward_validation(self, crash_recovery_prices):
        """Walk-forward should produce multiple periods."""
        import numpy as np
        from osrs_flipper.backtest import SignalBacktester

        # Need longer data for walk-forward
        extended = pd.DataFrame({
            "timestamp": list(range(300)),
            "mid_price": [1000 + 50 * np.sin(i * 0.1) + np.random.randn() * 20 for i in range(300)],
        })

        bt = SignalBacktester(extended, "Dragon bones")
        result = bt.walk_forward_test(
            train_days=60,
            test_days=30,
            hold_days=10,
            percentile_threshold=30,
        )

        assert "periods" in result
        assert "overall_win_rate" in result
        assert "overall_trades" in result
