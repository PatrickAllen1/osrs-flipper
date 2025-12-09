# OSRS Flipper

A Grand Exchange flip scanner and portfolio optimizer for Old School RuneScape.

## Features

- **Real-time Price Data**: Fetches live prices from the OSRS Wiki API
- **Technical Analysis**: RSI, percentile ranking, support/resistance detection
- **Pattern Detection**: Oversold opportunities, oscillator patterns
- **Monte Carlo Simulation**: Block bootstrap price path simulation with regime detection
- **Portfolio Optimization**: Pareto-efficient item selection balancing ROI vs risk
- **Vectorized Performance**: NumPy-optimized calculations (1M+ simulations/second)

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

### CLI Commands

```bash
# Scan for flip opportunities
osrs-flip scan

# Analyze a specific item
osrs-flip analyze 2

# Run Monte Carlo simulation
osrs-flip simulate 2 --sims 10000 --days 30

# Backtest strategy on historical data
osrs-flip backtest 2

# Build optimized portfolio
osrs-flip portfolio --top 20
```

### As a Library

```python
from osrs_flipper import OSRSClient, HistoryFetcher, MonteCarloSimulator

# Fetch current prices
client = OSRSClient()
prices = client.fetch_latest()

# Get historical data
fetcher = HistoryFetcher()
df = fetcher.get_dataframe(item_id=2, timestep="24h")

# Run simulation
simulator = MonteCarloSimulator(df["mid_price"])
results = simulator.run(n_sims=10000, n_days=30)
print(f"Expected ROI: {results['roi_percentiles']['50']}%")
```

## Architecture

```
osrs_flipper/
├── api.py          # OSRS Wiki API client
├── cache.py        # Historical data caching
├── history.py      # Data fetching with incremental updates
├── features.py     # Feature engineering (returns, volatility, etc.)
├── indicators.py   # Technical indicators (RSI, percentile)
├── analyzers.py    # Pattern detection (oversold, oscillator)
├── regimes.py      # Market regime classification
├── simulator.py    # Monte Carlo simulation engine
├── backtest.py     # Strategy backtesting
├── portfolio.py    # Pareto-optimal portfolio selection
├── scanner.py      # Opportunity scanner
└── cli.py          # Command-line interface
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT
