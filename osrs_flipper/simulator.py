"""Monte Carlo simulation utilities for backtesting."""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.signal import lfilter


def block_bootstrap_sample(
    returns: np.ndarray,
    n_samples: int,
    block_size: int = 5,
) -> np.ndarray:
    """Sample returns using block bootstrap.

    Preserves local autocorrelation by sampling contiguous blocks
    and concatenating them.

    Args:
        returns: Array of historical returns.
        n_samples: Number of samples to generate.
        block_size: Size of each block.

    Returns:
        Array of sampled returns with length n_samples.
    """
    # Handle empty data case
    if len(returns) == 0:
        return np.zeros(n_samples)

    # Adjust block_size if larger than data
    if block_size > len(returns):
        block_size = len(returns)

    # Calculate number of blocks needed
    n_blocks = int(np.ceil(n_samples / block_size))

    # Randomly select start indices for each block
    max_start_idx = len(returns) - block_size
    if max_start_idx <= 0:
        # If data is smaller than or equal to block_size,
        # we can only start at index 0
        start_indices = np.zeros(n_blocks, dtype=int)
    else:
        start_indices = np.random.randint(0, max_start_idx + 1, size=n_blocks)

    # Extract blocks and concatenate
    blocks = []
    for start_idx in start_indices:
        block = returns[start_idx : start_idx + block_size]
        blocks.append(block)

    sampled = np.concatenate(blocks)

    # Truncate to exactly n_samples
    return sampled[:n_samples]


def block_bootstrap_sample_batch(
    returns: np.ndarray,
    n_sims: int,
    n_samples: int,
    block_size: int = 5,
) -> np.ndarray:
    """Vectorized block bootstrap for multiple simulations.

    Args:
        returns: 1D array of historical returns
        n_sims: Number of simulations to generate
        n_samples: Samples per simulation
        block_size: Size of each contiguous block

    Returns:
        Array of shape (n_sims, n_samples) with bootstrapped returns
    """
    if len(returns) == 0:
        return np.zeros((n_sims, n_samples))

    # Adjust block_size if needed
    block_size = min(block_size, len(returns))
    n_blocks = int(np.ceil(n_samples / block_size))

    # Generate all start indices at once: shape (n_sims, n_blocks)
    max_start = max(0, len(returns) - block_size)
    start_indices = np.random.randint(0, max_start + 1, size=(n_sims, n_blocks))

    # Create offset array: shape (block_size,)
    offsets = np.arange(block_size)

    # Broadcasting: (n_sims, n_blocks, 1) + (block_size,) = (n_sims, n_blocks, block_size)
    indices = start_indices[:, :, np.newaxis] + offsets

    # Gather all blocks at once
    blocks = returns[indices]  # shape (n_sims, n_blocks, block_size)

    # Reshape to (n_sims, n_blocks * block_size) and truncate
    sampled = blocks.reshape(n_sims, -1)[:, :n_samples]

    return sampled


def generate_price_paths_batch(
    start_price: int,
    returns: np.ndarray,
    n_sims: int,
    n_days: int,
    block_size: int = 5,
    mean_reversion_strength: float = 0.0,
    historical_mean: Optional[float] = None,
    momentum_factor: float = 0.0,
    volatility_multiplier: float = 1.0,
) -> np.ndarray:
    """Generate multiple price paths in parallel.

    Args:
        start_price: Starting price in GP
        returns: Historical returns array
        n_sims: Number of simulations
        n_days: Days per simulation
        block_size: Block size for bootstrap
        mean_reversion_strength: Mean reversion parameter
        historical_mean: Target mean for reversion
        momentum_factor: Momentum carry-over factor
        volatility_multiplier: Volatility scaling

    Returns:
        Array of shape (n_sims, n_days + 1) with price paths
    """
    # Step 1: Bootstrap all returns at once
    sampled = block_bootstrap_sample_batch(returns, n_sims, n_days, block_size)
    sampled = sampled * volatility_multiplier  # shape (n_sims, n_days)

    # Step 2: Apply momentum via IIR filter if needed
    if momentum_factor != 0.0:
        # y[n] = x[n] + momentum_factor * y[n-1]
        # This is scipy.signal.lfilter with b=[1], a=[1, -momentum_factor]
        sampled = lfilter([1], [1, -momentum_factor], sampled, axis=1)

    # Step 3: Build price paths
    if mean_reversion_strength == 0.0:
        # Pure vectorized path generation
        multipliers = 1 + sampled  # shape (n_sims, n_days)
        cumulative = np.cumprod(multipliers, axis=1)

        # Prepend 1.0 for start_price multiplier
        ones = np.ones((n_sims, 1))
        cumulative = np.hstack([ones, cumulative])

        paths = (start_price * cumulative).astype(int)
        paths = np.maximum(paths, 1)  # Floor at 1 GP

    else:
        # Mean reversion requires sequential processing per day
        # But we vectorize across simulations
        paths = np.zeros((n_sims, n_days + 1), dtype=int)
        paths[:, 0] = start_price

        for day in range(n_days):
            current_prices = paths[:, day].astype(float)
            base_returns = sampled[:, day]

            # Mean reversion adjustment (vectorized across sims)
            reversion = mean_reversion_strength * (historical_mean - current_prices) / np.maximum(current_prices, 1)
            adjusted_returns = base_returns + reversion

            # Apply returns
            new_prices = current_prices * (1 + adjusted_returns)
            paths[:, day + 1] = np.maximum(1, new_prices.astype(int))

    return paths


def generate_price_path(
    start_price: int,
    returns: np.ndarray,
    n_days: int,
    block_size: int = 5,
    mean_reversion_strength: float = 0.0,
    historical_mean: Optional[float] = None,
    momentum_factor: float = 0.0,
    volatility_multiplier: float = 1.0,
) -> List[int]:
    """
    Generate a simulated price path using block bootstrap with regime adjustments.

    Args:
        start_price: Starting price in GP
        returns: Historical returns array
        n_days: Number of days to simulate
        block_size: Block size for bootstrap sampling
        mean_reversion_strength: How strongly to pull toward mean (0 = none)
        historical_mean: Mean price to revert to (required if mean_reversion_strength > 0)
        momentum_factor: How much previous return affects current (0 = none)
        volatility_multiplier: Scale the sampled volatility (1.0 = unchanged)

    Returns:
        List of integer prices [start_price, day1, day2, ..., dayN]
    """
    # Sample returns
    sampled = block_bootstrap_sample(returns, n_days, block_size)

    # Apply volatility multiplier
    sampled = sampled * volatility_multiplier

    # Generate path
    path = [start_price]
    prev_return = 0.0

    for i in range(n_days):
        base_return = sampled[i]

        # Add momentum effect
        adjusted_return = base_return + momentum_factor * prev_return

        # Add mean reversion effect
        if mean_reversion_strength > 0 and historical_mean:
            reversion = mean_reversion_strength * (historical_mean - path[-1]) / path[-1]
            adjusted_return += reversion

        # Apply return and floor at 1 GP
        new_price = max(1, int(path[-1] * (1 + adjusted_return)))
        path.append(new_price)

        prev_return = adjusted_return

    return path


class MonteCarloSimulator:
    """Run Monte Carlo simulations for price forecasting."""

    def __init__(self, prices: pd.Series, start_price: int = None):
        """
        Initialize simulator with historical prices.

        Args:
            prices: Historical price series
            start_price: Starting price for simulation (default: last price)
        """
        from .features import calculate_returns
        from .regimes import RegimeClassifier

        self.prices = prices
        self.start_price = start_price or int(prices.iloc[-1])
        self.returns = calculate_returns(prices).dropna().values
        # Use 60-day rolling mean to avoid bias from old prices
        # For declining items, the rolling mean also declines
        rolling_window = min(60, len(prices))
        self.historical_mean = float(prices.tail(rolling_window).mean())

        # Detect regime and get params
        self.classifier = RegimeClassifier()
        self.regime = self.classifier.classify(prices)
        self.sim_params = self.classifier.get_simulation_params(self.regime)

    def run(
        self,
        n_sims: int = 10000,
        n_days: int = 30,
        block_size: int = 5,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.

        Args:
            n_sims: Number of simulations
            n_days: Days to simulate
            block_size: Block size for bootstrap

        Returns:
            Dict with: start_price, prob_profit, prob_loss, expected_value,
            percentiles{5,25,50,75,95}, roi_percentiles, regime, n_sims, n_days
        """
        # Single vectorized call replaces n_sims iterations
        paths = generate_price_paths_batch(
            start_price=self.start_price,
            returns=self.returns,
            n_sims=n_sims,
            n_days=n_days,
            block_size=block_size,
            mean_reversion_strength=self.sim_params["mean_reversion_strength"],
            historical_mean=self.historical_mean,
            momentum_factor=self.sim_params["momentum_factor"],
            volatility_multiplier=self.sim_params["volatility_multiplier"],
        )

        final_prices = paths[:, -1]  # Extract final day prices

        # Calculate statistics
        profits = final_prices - self.start_price
        rois = (profits / self.start_price) * 100

        return {
            "start_price": self.start_price,
            "prob_profit": float(np.mean(profits > 0)),
            "prob_loss": float(np.mean(profits < 0)),
            "expected_value": int(np.mean(final_prices)),
            "percentiles": {
                "5": int(np.percentile(final_prices, 5)),
                "25": int(np.percentile(final_prices, 25)),
                "50": int(np.percentile(final_prices, 50)),
                "75": int(np.percentile(final_prices, 75)),
                "95": int(np.percentile(final_prices, 95)),
            },
            "roi_percentiles": {
                "5": round(float(np.percentile(rois, 5)), 2),
                "25": round(float(np.percentile(rois, 25)), 2),
                "50": round(float(np.percentile(rois, 50)), 2),
                "75": round(float(np.percentile(rois, 75)), 2),
                "95": round(float(np.percentile(rois, 95)), 2),
            },
            "regime": self.regime.value,
            "n_sims": n_sims,
            "n_days": n_days,
        }
