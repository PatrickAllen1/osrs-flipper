"""Market regime classification for OSRS GE items."""
from enum import Enum
from typing import Dict
import pandas as pd


class Regime(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    CHAOTIC = "chaotic"


class RegimeClassifier:
    """Classifies market regime for an item based on price history."""

    def __init__(
        self,
        trend_threshold: float = 0.02,
        volatility_threshold: float = 0.05,
        mean_reversion_threshold: float = 10,
    ):
        """Initialize regime classifier with thresholds.

        Args:
            trend_threshold: Threshold for average return to classify as trending (default 2%).
            volatility_threshold: Threshold for volatility to classify as chaotic (default 5%).
            mean_reversion_threshold: Threshold for half-life to classify as mean-reverting (default 10 days).
        """
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        self.mean_reversion_threshold = mean_reversion_threshold

    def classify(self, prices: pd.Series) -> Regime:
        """Classify market regime based on price series.

        Logic:
        1. Calculate average daily return over last 30 days
        2. Calculate volatility (std of returns)
        3. Estimate mean reversion half-life

        Classification:
        - If avg_return > trend_threshold: TRENDING_UP
        - If avg_return < -trend_threshold: TRENDING_DOWN
        - If volatility > volatility_threshold: CHAOTIC
        - If half_life < mean_reversion_threshold: MEAN_REVERTING
        - Default: MEAN_REVERTING

        Args:
            prices: Price series.

        Returns:
            Regime classification.
        """
        from .features import calculate_returns, calculate_volatility, estimate_mean_reversion_half_life

        # Calculate returns
        returns = calculate_returns(prices)
        avg_return = returns.tail(30).mean() if len(returns) >= 30 else returns.mean()

        # Calculate volatility
        volatility = calculate_volatility(prices, window=14)
        current_vol = volatility.iloc[-1] if len(volatility) > 0 else 0

        # Estimate mean reversion half-life
        half_life = estimate_mean_reversion_half_life(prices)

        # Classification logic - check volatility first as it overrides trends
        if current_vol > self.volatility_threshold:
            return Regime.CHAOTIC
        elif avg_return > self.trend_threshold:
            return Regime.TRENDING_UP
        elif avg_return < -self.trend_threshold:
            return Regime.TRENDING_DOWN
        elif half_life < self.mean_reversion_threshold:
            return Regime.MEAN_REVERTING
        else:
            return Regime.MEAN_REVERTING  # default

    def get_simulation_params(self, regime: Regime) -> Dict[str, float]:
        """Get simulation parameters for a given regime.

        Returns dict with: mean_reversion_strength, momentum_factor, volatility_multiplier

        Args:
            regime: Market regime.

        Returns:
            Dictionary of simulation parameters.
        """
        params = {
            Regime.TRENDING_UP: {
                "mean_reversion_strength": 0.0,
                "momentum_factor": 0.3,
                "volatility_multiplier": 1.0,
            },
            Regime.TRENDING_DOWN: {
                "mean_reversion_strength": 0.0,
                "momentum_factor": 0.3,
                "volatility_multiplier": 1.0,
            },
            Regime.MEAN_REVERTING: {
                "mean_reversion_strength": 0.1,
                "momentum_factor": 0.0,
                "volatility_multiplier": 1.0,
            },
            Regime.CHAOTIC: {
                "mean_reversion_strength": 0.05,
                "momentum_factor": 0.1,
                "volatility_multiplier": 1.5,
            },
        }
        return params[regime]
