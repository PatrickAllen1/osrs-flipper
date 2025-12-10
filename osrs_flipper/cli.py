"""Command-line interface for OSRS Flip Scanner."""
import os
from datetime import datetime
import click

from .api import OSRSClient
from .allocator import SlotAllocator
from .backtest import SignalBacktester
from .defaults import get_default_hold_days
from .history import HistoryFetcher
from .lookback import calculate_lookback_days
from .portfolio import PRESETS, PortfolioManager
from .scanner import ItemScanner
from .scoring import calculate_item_score
from .simulator import MonteCarloSimulator
from .tax import calculate_ge_tax, is_tax_exempt, GE_TAX_RATE
from .utils import parse_cash


@click.group()
@click.version_option(version="0.1.0")
def main():
    """OSRS Grand Exchange flip scanner and portfolio optimizer."""
    pass


@main.command()
@click.option(
    "--mode",
    type=click.Choice(["instant", "convergence", "both", "oversold", "oscillator", "all"]),
    default="both",
    help="Scanning mode: instant (same-day arbitrage), convergence (crash recovery), both (default), or legacy modes",
)
@click.option(
    "--cash",
    type=str,
    default=None,
    help="Cash stack to allocate (e.g., '120m', '1.5b')",
)
@click.option(
    "--slots",
    type=click.IntRange(1, 8),
    default=8,
    help="Available GE slots (1-8)",
)
@click.option(
    "--rotations",
    type=int,
    default=3,
    help="Buy limit rotations (default: 3)",
)
@click.option(
    "--strategy",
    type=click.Choice(["flip", "hold", "balanced"]),
    default="balanced",
    help="Allocation strategy",
)
@click.option(
    "--export",
    is_flag=True,
    help="Export results to CSV",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./output",
    help="Output directory for exports",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit items to scan (default: all ~4000)",
)
@click.option(
    "--hold-days",
    type=int,
    default=None,
    help="Expected hold period in days (default: based on strategy)",
)
@click.option(
    "--min-roi",
    type=float,
    default=20.0,
    help="Minimum tax-adjusted ROI % to include (default: 20)",
)
def scan(mode, cash, slots, rotations, strategy, export, output_dir, limit, hold_days, min_roi):
    """Scan for flip opportunities.

    Modes:
    - instant: High spread arbitrage opportunities (same-day flips)
    - convergence: Items crashed from recent highs (1-7 day recovery)
    - both: Find items matching either strategy (recommended)
    - oversold/oscillator/all: Legacy long-term strategies
    """
    click.echo(f"OSRS Flip Scanner - {mode.upper()} mode")
    click.echo("=" * 60)

    # Calculate hold_days from strategy if not provided
    if hold_days is None:
        hold_days = get_default_hold_days(strategy)

    # Calculate lookback window
    lookback_days = calculate_lookback_days(hold_days)

    click.echo(f"Hold time: {hold_days} days | Lookback: {lookback_days} days | Min ROI: {min_roi}%")

    client = OSRSClient()
    scanner = ItemScanner(client)

    click.echo("Fetching data...")

    # Progress callback for terminal feedback
    def progress(current, total):
        if current % 50 == 0 or current == total:
            click.echo(f"  Scanning items: {current}/{total}", nl=False)
            click.echo("\r", nl=False)

    opportunities = scanner.scan(
        mode=mode,
        limit=limit,
        progress_callback=progress,
        lookback_days=lookback_days,
        min_roi=min_roi,
    )
    click.echo()  # newline after progress

    click.echo(f"Found {len(opportunities)} opportunities")

    if not opportunities:
        click.echo("No opportunities found matching criteria.")
        return

    # Add scores
    for opp in opportunities:
        oversold = opp.get("oversold", {})
        opp["score"] = calculate_item_score(
            upside_pct=oversold.get("upside_pct", 0),
            percentile=oversold.get("percentile", 50),
            volume_ratio=1.0,
            bounce_rate=0.5,
        )

    # Sort by score
    opportunities.sort(key=lambda x: x["score"], reverse=True)

    if cash:
        cash_gp = parse_cash(cash)
        click.echo(f"\nAllocating {cash_gp:,} GP across {slots} slots...")

        allocator = SlotAllocator(strategy=strategy)
        allocation = allocator.allocate(
            opportunities=opportunities,
            cash=cash_gp,
            slots=slots,
            rotations=rotations,
        )

        _print_allocation(allocation, strategy)

        if export:
            _export_allocation(allocation, output_dir)
    else:
        _print_opportunities(opportunities[:20])

        if export:
            _export_opportunities(opportunities, output_dir)


def _print_opportunities(opportunities):
    """Print opportunity list."""
    click.echo(f"\n{'Item':<30} {'Price':>10} {'%ile':>6} {'RSI':>5} {'Upside':>8} {'Score':>6}")
    click.echo("-" * 70)

    for opp in opportunities:
        oversold = opp.get("oversold", {})
        rsi = oversold.get("rsi")
        rsi_str = f"{rsi:.0f}" if rsi else "N/A"

        click.echo(
            f"{opp['name']:<30} "
            f"{opp['current_price']:>10,} "
            f"{oversold.get('percentile', 0):>5.1f}% "
            f"{rsi_str:>5} "
            f"{oversold.get('upside_pct', 0):>7.1f}% "
            f"{opp.get('score', 0):>6.1f}"
        )


def _momentum_arrow(momentum: float) -> str:
    """Convert momentum to visual indicator."""
    if momentum >= 1.5:
        return "↑↑"  # Strong buyers
    elif momentum >= 1.0:
        return "↑"   # Buyers winning
    elif momentum >= 0.7:
        return "→"   # Neutral
    else:
        return "↓"   # Sellers winning


def _print_allocation(allocation, strategy):
    """Print allocation table with selection reasons and exit strategies."""
    click.echo(f"\nAllocation ({strategy}):")
    click.echo(f"{'Slot':<4} {'Item':<22} {'Buy':>10} {'Target':>10} {'ROI':>6} {'Hold':>5} {'Mom':>4} {'Why':<12}")
    click.echo("-" * 85)

    total_capital = 0
    total_expected_profit = 0

    for slot in allocation:
        momentum = slot.get("buyer_momentum", 0)
        mom_str = f"{_momentum_arrow(momentum)}{momentum:.1f}"

        # Build "why" string
        pct = slot.get("percentile", 0)
        upside = slot.get("upside_pct", 0)
        why = f"{pct:.0f}%ile +{upside:.0f}%"

        hold_days = slot.get("expected_hold_days", 0)
        hold_str = f"{hold_days:.0f}d" if hold_days else "?"

        roi = slot.get("target_roi_pct", 0)

        click.echo(
            f"{slot['slot']:<4} "
            f"{slot['name'][:21]:<22} "
            f"{slot['buy_price']:>10,} "
            f"{slot.get('target_price', 0):>10,} "
            f"{roi:>5.1f}% "
            f"{hold_str:>5} "
            f"{mom_str:>4} "
            f"{why:<12}"
        )
        total_capital += slot["capital"]
        expected_profit = slot["capital"] * (roi / 100)
        total_expected_profit += expected_profit

    click.echo("-" * 85)
    click.echo(f"{'Total Capital:':<30} {total_capital:>12,} GP")
    click.echo(f"{'Expected Profit (target):':<30} {int(total_expected_profit):>12,} GP")

    # Print exit strategy legend
    click.echo(f"\n{'Exit Strategy:'}")
    click.echo(f"  Target = 50th percentile (median historical price)")
    click.echo(f"  Momentum: ↑↑ strong buy pressure, ↑ buyers winning, → neutral, ↓ sellers winning")


def _export_allocation(allocation, output_dir):
    """Export allocation to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    filepath = os.path.join(output_dir, f"allocation_{timestamp}.csv")

    with open(filepath, "w") as f:
        f.write("slot,item,item_id,buy_price,quantity,capital,buy_limit,rotations\n")
        for slot in allocation:
            f.write(
                f"{slot['slot']},{slot['name']},{slot.get('item_id', '')},"
                f"{slot['buy_price']},{slot['quantity']},{slot['capital']},"
                f"{slot['buy_limit']},{slot['rotations']}\n"
            )

    click.echo(f"\nExported to: {filepath}")


def _export_opportunities(opportunities, output_dir):
    """Export opportunities to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    filepath = os.path.join(output_dir, f"scan_{timestamp}.csv")

    with open(filepath, "w") as f:
        f.write("item,item_id,price,percentile,rsi,upside_pct,volume,score\n")
        for opp in opportunities:
            oversold = opp.get("oversold", {})
            f.write(
                f"{opp['name']},{opp['item_id']},{opp['current_price']},"
                f"{oversold.get('percentile', '')},"
                f"{oversold.get('rsi', '')},"
                f"{oversold.get('upside_pct', '')},"
                f"{opp['daily_volume']},"
                f"{opp.get('score', '')}\n"
            )

    click.echo(f"\nExported to: {filepath}")


@main.command()
@click.option(
    "--list",
    "list_presets",
    is_flag=True,
    help="List saved portfolio presets",
)
@click.option(
    "--use",
    type=str,
    default=None,
    help="Use a preset (grinder, balanced, diamondhands)",
)
@click.option(
    "--recommend",
    is_flag=True,
    help="Auto-recommend portfolio based on market",
)
@click.option(
    "--cash",
    type=str,
    default=None,
    help="Cash stack to allocate",
)
@click.option(
    "--slots",
    type=click.IntRange(1, 8),
    default=8,
    help="Available GE slots (1-8)",
)
@click.option(
    "--save",
    type=str,
    default=None,
    help="Save current allocation as preset",
)
@click.option(
    "--optimize",
    is_flag=True,
    help="Rank opportunities using Monte Carlo simulation",
)
@click.option(
    "--sims",
    type=int,
    default=1000,
    help="Number of simulations per item (default: 1000)",
)
@click.pass_context
def portfolio(ctx, list_presets, use, recommend, cash, slots, save, optimize, sims):
    """Manage portfolio presets and recommendations."""
    manager = PortfolioManager()

    if list_presets:
        click.echo("Available presets:")
        click.echo("-" * 50)
        for name, preset in PRESETS.items():
            click.echo(f"  {name:<15} - {preset['description']}")
        return

    if recommend:
        if not cash:
            raise click.UsageError("--cash required with --recommend")

        click.echo("Analyzing market conditions...")

        client = OSRSClient()
        scanner = ItemScanner(client)
        opportunities = scanner.scan(mode="all")

        recommendation = manager.recommend(opportunities)
        preset = manager.get_preset(recommendation)

        click.echo(f"\nRecommendation: {recommendation}")
        click.echo(f"  {preset['description']}")
        click.echo(f"\nRun: osrs-flip scan --cash {cash} --strategy {_preset_to_strategy(recommendation)}")
        return

    if use:
        if not cash:
            raise click.UsageError("--cash required with --use")

        preset = manager.get_preset(use)
        if not preset:
            raise click.BadParameter(f"Unknown preset: {use}")

        click.echo(f"Loading preset: {use}")
        click.echo(f"  {preset['description']}")

        strategy = _preset_to_strategy(use)

        ctx.invoke(
            scan,
            mode="all",
            cash=cash,
            slots=slots,
            strategy=strategy,
            rotations=3,
            export=False,
            output_dir="./output",
        )
        return

    if optimize:
        click.echo("Portfolio Optimization Mode")
        click.echo("=" * 60)

        # Scan for opportunities
        client = OSRSClient()
        scanner = ItemScanner(client)
        fetcher = HistoryFetcher(client=client)

        click.echo("Scanning for opportunities...")
        opportunities = scanner.scan(mode="all")

        if not opportunities:
            click.echo("No opportunities found.")
            return

        click.echo(f"Found {len(opportunities)} opportunities")
        click.echo(f"Running Monte Carlo simulations ({sims} sims per item)...")
        click.echo()

        # Score each opportunity with Monte Carlo simulation
        scored_opportunities = []
        for i, opp in enumerate(opportunities):
            try:
                # Get historical data
                df = fetcher.get_dataframe(opp["item_id"], timestep="24h")

                if df.empty or len(df) < 7:
                    continue

                # Run simulation
                simulator = MonteCarloSimulator(
                    df["mid_price"],
                    start_price=opp["current_price"]
                )
                results = simulator.run(n_sims=sims, n_days=30)

                # Calculate score: prob_profit * median_roi
                prob_profit = results["prob_profit"]
                median_roi = results["roi_percentiles"]["50"]
                score = prob_profit * median_roi

                scored_opportunities.append({
                    "name": opp["name"],
                    "score": score,
                    "prob_profit": prob_profit,
                    "median_roi": median_roi,
                })

                # Progress feedback
                if (i + 1) % 10 == 0 or (i + 1) == len(opportunities):
                    click.echo(f"  Processed: {i + 1}/{len(opportunities)}", nl=False)
                    click.echo("\r", nl=False)

            except Exception:
                # Skip items that fail simulation
                continue

        click.echo()

        if not scored_opportunities:
            click.echo("No items had sufficient data for simulation.")
            return

        # Sort by score descending
        scored_opportunities.sort(key=lambda x: x["score"], reverse=True)

        # Display top opportunities
        click.echo(f"\nTop Opportunities (sorted by score):")
        click.echo(f"{'Item':<30} {'Score':>10} {'Prob Profit':>12} {'Median ROI':>12}")
        click.echo("-" * 70)

        for opp in scored_opportunities[:20]:
            click.echo(
                f"{opp['name']:<30} "
                f"{opp['score']:>10.2f} "
                f"{opp['prob_profit'] * 100:>11.1f}% "
                f"{opp['median_roi']:>11.1f}%"
            )

        return

    click.echo("Use --list, --use <preset>, --recommend, or --optimize")


def _preset_to_strategy(preset_name: str) -> str:
    """Map preset name to allocation strategy."""
    preset = PRESETS.get(preset_name, {})
    if preset.get("flip_ratio", 0) > 0.6:
        return "flip"
    elif preset.get("hold_ratio", 0) > 0.6:
        return "hold"
    return "balanced"


@main.command()
@click.argument("item_name", type=str)
@click.option(
    "--days",
    type=int,
    default=30,
    help="Simulation horizon in days (default: 30)",
)
@click.option(
    "--sims",
    type=int,
    default=10000,
    help="Number of Monte Carlo simulations (default: 10000)",
)
@click.option(
    "--entry",
    type=str,
    default=None,
    help="Your entry price (e.g., '1.5m', '500k'). Analyzes existing position.",
)
def deep(item_name, days, sims, entry):
    """Run Monte Carlo analysis on a specific item."""
    # Parse entry price if provided
    entry_price = None
    if entry:
        entry_price = parse_cash(entry)
        click.echo(f"Position Analysis: {item_name}")
        click.echo(f"Entry Price: {entry_price:,} GP")
    else:
        click.echo(f"Deep Analysis: {item_name}")
    click.echo("=" * 60)

    # Initialize services
    client = OSRSClient()
    fetcher = HistoryFetcher(client=client)

    click.echo("Fetching item data...")

    # Get item ID from mapping
    mapping = client.fetch_mapping()
    item_id = None
    for id_key, item_data in mapping.items():
        name = item_data.get("name", "") if isinstance(item_data, dict) else str(item_data)
        if name.lower() == item_name.lower():
            item_id = int(id_key)
            break

    if item_id is None:
        click.echo(f"Error: Item '{item_name}' not found.", err=True)
        return

    # Get latest price
    all_latest = client.fetch_latest()
    latest = all_latest.get(str(item_id), {})
    if not latest or "high" not in latest:
        click.echo(f"Error: Could not fetch current price for {item_name}.", err=True)
        return

    current_price = latest["high"]

    click.echo(f"Fetching 6-month price history...")

    # Get historical data
    df = fetcher.get_dataframe(item_id, timestep="24h")

    if df.empty or len(df) < 7:
        click.echo(f"Error: Insufficient historical data for {item_name}.", err=True)
        return

    # Calculate 6-month range (last 180 days)
    prices_180d = df["mid_price"].tail(180)
    price_low = int(prices_180d.min())
    price_high = int(prices_180d.max())

    click.echo(f"\nRunning Monte Carlo simulation ({sims:,} sims, {days} days)...")

    # Run Monte Carlo simulation from current price
    simulator = MonteCarloSimulator(df["mid_price"], start_price=current_price)
    results = simulator.run(n_sims=sims, n_days=days)

    # Tax setup
    tax_exempt = is_tax_exempt(item_name)
    tax_mult = 1.0 if tax_exempt else (1 - GE_TAX_RATE)

    # Display results
    click.echo(f"\n{'Item Information':}")
    click.echo(f"  Name:          {item_name}")
    click.echo(f"  Current Price: {current_price:,} GP")

    # Show position info if entry price provided
    if entry_price:
        unrealized_pnl = current_price - entry_price
        unrealized_pct = (unrealized_pnl / entry_price) * 100
        # If sold now, what would the after-tax P&L be?
        tax_if_sold_now = calculate_ge_tax(current_price, item_name)
        realized_pnl_now = current_price - tax_if_sold_now - entry_price
        realized_pct_now = (realized_pnl_now / entry_price) * 100

        click.echo(f"\n{'Your Position':}")
        click.echo(f"  Entry Price:   {entry_price:,} GP")
        click.echo(f"  Unrealized:    {unrealized_pnl:+,} GP ({unrealized_pct:+.1f}%)")
        click.echo(f"  If Sold Now:   {realized_pnl_now:+,} GP ({realized_pct_now:+.1f}%) after tax")

    click.echo(f"\n{'6-Month Price Range':}")
    click.echo(f"  Low:           {price_low:,} GP")
    click.echo(f"  High:          {price_high:,} GP")
    click.echo(f"\n{'Detected Regime':}")
    click.echo(f"  {results['regime']}")

    # Probability analysis - recalculate based on entry price if provided
    if entry_price:
        # Recalculate probabilities based on entry price, not current price
        # We need to determine what % of simulated outcomes are profitable vs entry
        # The simulation gives us price outcomes, we convert to P&L vs entry
        click.echo(f"\n{'Probability Analysis (vs your entry)':}")

        # Calculate breakeven sell price (entry + tax)
        # breakeven: sell * tax_mult = entry → sell = entry / tax_mult
        breakeven_sell = int(entry_price / tax_mult)
        click.echo(f"  Breakeven:     {breakeven_sell:,} GP (to recover entry after tax)")

        # For each percentile outcome, calculate profit vs entry
        for pct_key in ['5', '25', '50', '75', '95']:
            sim_price = results['percentiles'][pct_key]
            tax_at_price = calculate_ge_tax(sim_price, item_name)
            net_vs_entry = sim_price - tax_at_price - entry_price
            roi_vs_entry = (net_vs_entry / entry_price) * 100
            results['roi_percentiles'][pct_key] = round(roi_vs_entry, 2)

        # Estimate profit probability vs entry
        # Approximate: if median outcome > breakeven, more likely profit
        median_outcome = results['percentiles']['50']
        if median_outcome >= breakeven_sell:
            # Rough interpolation
            p25 = results['percentiles']['25']
            if p25 >= breakeven_sell:
                est_prob_profit = 0.75 + 0.25 * (median_outcome - breakeven_sell) / max(1, median_outcome - p25)
            else:
                est_prob_profit = 0.50 + 0.25 * (median_outcome - breakeven_sell) / max(1, median_outcome - p25)
        else:
            p75 = results['percentiles']['75']
            if p75 >= breakeven_sell:
                est_prob_profit = 0.25 + 0.25 * (p75 - breakeven_sell) / max(1, p75 - median_outcome)
            else:
                est_prob_profit = 0.25 * (results['percentiles']['95'] - breakeven_sell) / max(1, results['percentiles']['95'] - p75)
        est_prob_profit = max(0, min(1, est_prob_profit))

        click.echo(f"  Profit:        ~{est_prob_profit * 100:.0f}% (estimated vs entry)")
        click.echo(f"  Loss:          ~{(1 - est_prob_profit) * 100:.0f}%")
    else:
        click.echo(f"\n{'Probability Analysis':}")
        click.echo(f"  Profit:        {results['prob_profit'] * 100:.1f}%")
        click.echo(f"  Loss:          {results['prob_loss'] * 100:.1f}%")

    click.echo(f"\n{'Price Outcomes (GP)':}")
    click.echo(f"  5th %ile:      {results['percentiles']['5']:,} GP")
    click.echo(f"  25th %ile:     {results['percentiles']['25']:,} GP")
    click.echo(f"  50th %ile:     {results['percentiles']['50']:,} GP (median)")
    click.echo(f"  75th %ile:     {results['percentiles']['75']:,} GP")
    click.echo(f"  95th %ile:     {results['percentiles']['95']:,} GP")

    # ROI section header depends on whether we have entry price
    if entry_price:
        click.echo(f"\n{'ROI vs Entry (after tax)':}")
    else:
        click.echo(f"\n{'ROI Outcomes':}")
    click.echo(f"  5th %ile:      {results['roi_percentiles']['5']:.1f}%")
    click.echo(f"  25th %ile:     {results['roi_percentiles']['25']:.1f}%")
    click.echo(f"  50th %ile:     {results['roi_percentiles']['50']:.1f}% (median)")
    click.echo(f"  75th %ile:     {results['roi_percentiles']['75']:.1f}%")
    click.echo(f"  95th %ile:     {results['roi_percentiles']['95']:.1f}%")

    # Sell targets - use entry price as cost basis if provided
    cost_basis = entry_price if entry_price else current_price
    click.echo(f"\n{'Sell Targets (after 2% GE tax)':}")
    if tax_exempt:
        click.echo(f"  Note: {item_name} is TAX EXEMPT")

    for pct_label, roi in [
        ("5th %ile", results['roi_percentiles']['5']),
        ("25th %ile", results['roi_percentiles']['25']),
        ("50th %ile", results['roi_percentiles']['50']),
        ("75th %ile", results['roi_percentiles']['75']),
        ("95th %ile", results['roi_percentiles']['95']),
    ]:
        # sell = cost_basis * (1 + ROI) / tax_mult
        target_roi_decimal = roi / 100
        sell_price = int(cost_basis * (1 + target_roi_decimal) / tax_mult)
        tax_amount = calculate_ge_tax(sell_price, item_name)
        net_profit = sell_price - tax_amount - cost_basis
        suffix = " (median)" if "50th" in pct_label else ""
        click.echo(
            f"  {pct_label}:      {sell_price:,} GP "
            f"(tax: {tax_amount:,}, net: {net_profit:+,}){suffix}"
        )


@main.command()
@click.argument("item_name", type=str)
@click.option(
    "--train-days",
    type=int,
    default=90,
    help="Training window size in days (default: 90)",
)
@click.option(
    "--test-days",
    type=int,
    default=30,
    help="Test window size in days (default: 30)",
)
@click.option(
    "--hold-days",
    type=int,
    default=14,
    help="Hold period for trades in days (default: 14)",
)
@click.option(
    "--percentile",
    type=int,
    default=20,
    help="Entry threshold percentile (default: 20)",
)
def backtest(item_name, train_days, test_days, hold_days, percentile):
    """Run walk-forward validation backtest on a specific item."""
    click.echo(f"Backtest Analysis: {item_name}")
    click.echo("=" * 60)

    # Initialize services
    client = OSRSClient()
    fetcher = HistoryFetcher(client=client)

    click.echo("Fetching item data...")

    # Get item ID from mapping
    mapping = client.fetch_mapping()
    item_id = None
    for id_key, item_data in mapping.items():
        name = item_data.get("name", "") if isinstance(item_data, dict) else str(item_data)
        if name.lower() == item_name.lower():
            item_id = int(id_key)
            break

    if item_id is None:
        click.echo(f"Error: Item '{item_name}' not found.", err=True)
        return

    click.echo(f"Fetching historical price data...")

    # Get historical data (daily)
    df = fetcher.get_dataframe(item_id, timestep="24h")

    if df.empty or len(df) < train_days + test_days:
        click.echo(
            f"Error: Insufficient historical data for {item_name}. "
            f"Need at least {train_days + test_days} days, got {len(df)}.",
            err=True,
        )
        return

    click.echo(
        f"\nRunning walk-forward backtest "
        f"(train={train_days}d, test={test_days}d, hold={hold_days}d, "
        f"percentile={percentile})..."
    )

    # Run walk-forward backtest
    backtester = SignalBacktester(df, item_name=item_name)
    results = backtester.walk_forward_test(
        train_days=train_days,
        test_days=test_days,
        hold_days=hold_days,
        percentile_threshold=percentile,
    )

    # Display overall results
    click.echo(f"\n{'Overall Results':}")
    click.echo(f"  Total Trades:     {results['overall_trades']}")
    click.echo(f"  Win Rate:         {results['overall_win_rate'] * 100:.1f}%")
    click.echo(f"  Avg Return:       {results['overall_avg_return']:.2f}%")

    # Display period-by-period breakdown
    if results["periods"]:
        click.echo(f"\n{'Period-by-Period Breakdown':}")
        click.echo(
            f"{'Period':<8} {'Trades':>7} {'Win Rate':>10} {'Avg Return':>12}"
        )
        click.echo("-" * 40)
        for i, period in enumerate(results["periods"], 1):
            win_rate_pct = period["win_rate"] * 100 if period["win_rate"] else 0
            click.echo(
                f"{i:<8} {period['num_trades']:>7} "
                f"{win_rate_pct:>9.1f}% {period['avg_return']:>11.2f}%"
            )


if __name__ == "__main__":
    main()
