# OSRS Flipper

A Grand Exchange flip scanner and portfolio optimizer for Old School RuneScape.

## Features

### Trading Strategies
- **Instant Arbitrage**: Same-day flips on high spread items with strong buyer/seller ratio
- **Convergence Plays**: Items crashed from 1d/1w/1m highs for mean reversion (1-7 day holds)
- **Legacy Strategies**: Oversold detection and oscillator patterns for longer-term holds

### Analytics
- **Real-time Price Data**: Fetches live prices from the OSRS Wiki API
- **Multi-Timeframe Analysis**: Compare current prices to 1d, 1w, 1m highs
- **Technical Analysis**: RSI, percentile ranking, support/resistance detection
- **Monte Carlo Simulation**: Block bootstrap price path simulation with regime detection
- **Portfolio Optimization**: Pareto-efficient item selection balancing ROI vs risk
- **Tax Calculation**: Accurate GE tax calculations (2% for items >100gp, max 5M cap)
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

### Quick Start

```bash
# Scan for instant arbitrage + convergence opportunities (default mode)
python3 -m osrs_flipper.cli scan

# Scan instant arbitrage only (same-day flips)
python3 -m osrs_flipper.cli scan --mode instant --min-roi 10

# Scan convergence plays only (crash recovery)
python3 -m osrs_flipper.cli scan --mode convergence --min-roi 20

# Scan with portfolio allocation
python3 -m osrs_flipper.cli scan --cash 100m --slots 8 --strategy balanced

# Export results to CSV
python3 -m osrs_flipper.cli scan --export --output-dir ./results
```

### CLI Commands

#### `scan` - Find flip opportunities

**Modes:**
- `instant`: High spread arbitrage (same-day flips, 5%+ spread, BSR >1.2)
- `convergence`: Items crashed from recent highs (10%+ from 1d, 15%+ from 1w, 20%+ from 1m)
- `both`: Run both strategies simultaneously (default, recommended)
- `oversold`, `oscillator`, `all`: Legacy long-term strategies

**Key Options:**
```bash
--mode [instant|convergence|both|oversold|oscillator|all]  # Strategy mode (default: both)
--min-roi FLOAT                    # Minimum ROI % filter (default: 20)
--limit INTEGER                    # Max items to scan (default: all ~4000)
--cash TEXT                        # Allocate cash (e.g., '100m', '1.5b')
--slots INTEGER                    # GE slots available 1-8 (default: 8)
--strategy [flip|hold|balanced]    # Allocation strategy (default: balanced)
--hold-days INTEGER                # Expected hold period (auto-calculated from strategy)
--export                           # Export to CSV
--output-dir PATH                  # Output directory (default: ./output)
```

**Examples:**
```bash
# Quick instant flip scan
python3 -m osrs_flipper.cli scan --mode instant --min-roi 5 --limit 500

# Convergence plays with 30% minimum upside
python3 -m osrs_flipper.cli scan --mode convergence --min-roi 30

# Both strategies with allocation
python3 -m osrs_flipper.cli scan --mode both --cash 120m --slots 8 --strategy flip

# Legacy oversold detection
python3 -m osrs_flipper.cli scan --mode oversold --hold-days 14 --min-roi 25
```

#### `deep` - Monte Carlo analysis on specific item

```bash
# Analyze "Twisted bow" with 10k simulations
python3 -m osrs_flipper.cli deep "Twisted bow" --sims 10000 --days 30

# Analyze existing position
python3 -m osrs_flipper.cli deep "Scythe of vitur" --entry 650m --sims 10000
```

**Options:**
- `--sims INTEGER`: Number of simulations (default: 10000)
- `--days INTEGER`: Simulation horizon (default: 30)
- `--entry TEXT`: Your entry price for position analysis (e.g., '1.5m')

#### `backtest` - Walk-forward validation

```bash
# Backtest oversold strategy on "Dragon claws"
python3 -m osrs_flipper.cli backtest "Dragon claws" --hold-days 14 --percentile 20
```

**Options:**
- `--train-days INTEGER`: Training window (default: 90)
- `--test-days INTEGER`: Test window (default: 30)
- `--hold-days INTEGER`: Hold period (default: 14)
- `--percentile INTEGER`: Entry threshold (default: 20)

#### `portfolio` - Portfolio management

```bash
# List available presets
python3 -m osrs_flipper.cli portfolio --list

# Use preset
python3 -m osrs_flipper.cli portfolio --use grinder --cash 100m --slots 8

# Get recommendation
python3 -m osrs_flipper.cli portfolio --recommend --cash 100m

# Optimize with Monte Carlo
python3 -m osrs_flipper.cli portfolio --optimize --sims 1000
```

**Options:**
- `--list`: Show available presets (grinder, balanced, diamondhands)
- `--use TEXT`: Use a preset
- `--recommend`: Auto-recommend based on market conditions
- `--optimize`: Rank opportunities using Monte Carlo simulation
- `--sims INTEGER`: Simulations per item (default: 1000)

### As a Library

```python
from osrs_flipper import OSRSClient, ItemScanner

# Fetch opportunities
client = OSRSClient()
scanner = ItemScanner(client)

# Instant arbitrage scan
instant_opps = scanner.scan(mode="instant", min_roi=10, limit=100)
for opp in instant_opps:
    print(f"{opp['name']}: {opp['instant']['instant_roi_after_tax']:.1f}% ROI")

# Convergence scan
conv_opps = scanner.scan(mode="convergence", min_roi=20, limit=100)
for opp in conv_opps:
    conv = opp['convergence']
    print(f"{opp['name']}: {conv['upside_pct']:.1f}% upside to {conv['target_price']:,} GP")

# Monte Carlo simulation
from osrs_flipper import HistoryFetcher, MonteCarloSimulator

fetcher = HistoryFetcher(client=client)
df = fetcher.get_dataframe(item_id=2, timestep="24h")

simulator = MonteCarloSimulator(df["mid_price"], start_price=df["mid_price"].iloc[-1])
results = simulator.run(n_sims=10000, n_days=30)
print(f"Expected ROI: {results['roi_percentiles']['50']:.1f}%")
print(f"Profit Probability: {results['prob_profit']*100:.1f}%")
```

## Architecture

```
osrs_flipper/
├── api.py                  # OSRS Wiki API client
├── cache.py                # Historical data caching
├── history.py              # Data fetching with incremental updates
├── scanner.py              # Opportunity scanner (orchestrator)
│
├── Instant Arbitrage:
│   ├── spreads.py          # Spread % and tax-adjusted ROI calculations
│   ├── bsr.py              # Buyer/seller ratio (BSR) calculation
│   └── instant_analyzer.py # Instant spread opportunity detector
│
├── Convergence Strategy:
│   ├── timeframes.py           # Multi-timeframe high fetcher (1d/1w/1m)
│   └── convergence_analyzer.py # Mean reversion opportunity detector
│
├── Legacy Strategies:
│   ├── analyzers.py        # Pattern detection (oversold, oscillator)
│   ├── indicators.py       # Technical indicators (RSI, percentile)
│   └── exits.py            # Exit strategy calculation
│
├── Simulation & Backtest:
│   ├── features.py         # Feature engineering (returns, volatility, etc.)
│   ├── regimes.py          # Market regime classification
│   ├── simulator.py        # Monte Carlo simulation engine
│   └── backtest.py         # Strategy backtesting
│
├── Portfolio Management:
│   ├── allocator.py        # Slot allocation logic
│   ├── portfolio.py        # Pareto-optimal portfolio selection
│   └── scoring.py          # Item scoring
│
├── Utilities:
│   ├── tax.py              # GE tax calculation
│   ├── filters.py          # Volume filtering
│   ├── defaults.py         # Strategy defaults (hold days, min ROI)
│   ├── lookback.py         # Lookback window calculation
│   └── utils.py            # General utilities
│
└── cli.py                  # Command-line interface
```

## Strategy Details

### Instant Arbitrage
**Goal**: Same-day flips on high spread items with strong buyer demand

**Criteria**:
- Instant spread ≥ 5% (instasell - instabuy) / instabuy
- BSR ≥ 1.2 (buyers dominate sellers)
- Spread ≤ 25% (avoid suspicious outliers)
- Tax-adjusted ROI calculated using 2% GE tax

**ROI Formula**: `(instasell - tax - instabuy) / instabuy × 100`

### Convergence Plays
**Goal**: Items crashed from recent highs, positioned for mean reversion

**Criteria**:
- ≥10% below 1-day high
- ≥15% below 1-week high
- ≥20% below 1-month high
- BSR ≥ 0.8 (not being heavily dumped)

**Target**: Maximum of 1d/1w/1m highs
**Upside**: Distance from current price to target

### Legacy Strategies
- **Oversold**: Items in bottom 20th percentile with 30%+ upside to 6-month high
- **Oscillator**: Items bouncing between support/resistance, currently near support

## Testing

```bash
pytest tests/ -v
```

## License

MIT
