"""Tests for CLI interface."""
from click.testing import CliRunner
from osrs_flipper.cli import main, scan, portfolio


def test_cli_main_shows_help():
    """Main command shows help."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "OSRS Grand Exchange" in result.output


def test_scan_command_exists():
    """Scan subcommand is available."""
    runner = CliRunner()
    result = runner.invoke(scan, ["--help"])

    assert result.exit_code == 0
    assert "--mode" in result.output
    assert "--cash" in result.output
    assert "--slots" in result.output


def test_portfolio_command_exists():
    """Portfolio subcommand is available."""
    runner = CliRunner()
    result = runner.invoke(portfolio, ["--help"])

    assert result.exit_code == 0
    assert "--list" in result.output
    assert "--use" in result.output
    assert "--recommend" in result.output


def test_deep_command_exists():
    """Deep analysis command is available."""
    runner = CliRunner()
    result = runner.invoke(main, ["deep", "--help"])

    assert result.exit_code == 0
    assert "item_name" in result.output or "ITEM_NAME" in result.output


def test_backtest_command_exists():
    """Backtest subcommand is available."""
    runner = CliRunner()
    result = runner.invoke(main, ["backtest", "--help"])

    assert result.exit_code == 0
    assert "item_name" in result.output or "ITEM_NAME" in result.output


def test_portfolio_optimize_option():
    """Portfolio command has --optimize flag."""
    runner = CliRunner()
    result = runner.invoke(portfolio, ["--help"])

    assert result.exit_code == 0
    assert "--optimize" in result.output
    assert "--sims" in result.output


def test_scan_instant_mode():
    """CLI supports --mode instant."""
    runner = CliRunner()
    result = runner.invoke(scan, ["--mode", "instant", "--limit", "1"])

    # Should not error on mode
    assert "Invalid value for '--mode'" not in result.output


def test_scan_convergence_mode():
    """CLI supports --mode convergence."""
    runner = CliRunner()
    result = runner.invoke(scan, ["--mode", "convergence", "--limit", "1"])

    assert "Invalid value for '--mode'" not in result.output


def test_scan_both_mode():
    """CLI supports --mode both."""
    runner = CliRunner()
    result = runner.invoke(scan, ["--mode", "both", "--limit", "1"])

    assert "Invalid value for '--mode'" not in result.output
