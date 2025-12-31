#!/usr/bin/env python
"""
Run backtests for PolyBot strategies.

Usage:
    python scripts/run_backtest.py --strategy arbitrage --days 30
    python scripts/run_backtest.py --strategy ai_analyst --days 14
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from polybot.backtest import BacktestEngine, BacktestConfig
from polybot.strategies import ArbitrageStrategy, AIAnalystStrategy


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


async def run_backtest(
    strategy_name: str,
    days: int,
    initial_balance: float,
    position_size_pct: float,
):
    """Run a backtest."""
    logger = logging.getLogger(__name__)

    # Create strategy instance
    if strategy_name == "arbitrage":
        strategy = ArbitrageStrategy()
    elif strategy_name == "ai_analyst":
        strategy = AIAnalystStrategy()
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return

    # Configure backtest
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    config = BacktestConfig(
        strategy_name=strategy_name,
        start_date=start_date,
        end_date=end_date,
        initial_balance=initial_balance,
        position_size_pct=position_size_pct,
        max_positions=10,
        slippage_pct=0.1,
        stop_loss_pct=10.0,
        take_profit_pct=20.0,
    )

    logger.info(f"Running backtest: {strategy_name}")
    logger.info(f"  Period: {start_date.date()} to {end_date.date()} ({days} days)")
    logger.info(f"  Initial balance: ${initial_balance:,.2f}")
    logger.info(f"  Position size: {position_size_pct}%")

    # Run backtest
    engine = BacktestEngine()
    metrics = await engine.run(config, strategy)

    # Print results
    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS: {strategy_name}")
    print("=" * 60)
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Balance: ${metrics.initial_balance:,.2f}")
    print(f"Final Balance: ${metrics.final_balance:,.2f}")
    print(f"Total Return: ${metrics.total_return:,.2f} ({metrics.total_return_pct:.2f}%)")
    print("-" * 60)
    print(f"Total Trades: {metrics.total_trades}")
    print(f"Winning Trades: {metrics.winning_trades}")
    print(f"Losing Trades: {metrics.losing_trades}")
    print(f"Win Rate: {metrics.win_rate:.1f}%")
    print("-" * 60)
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print("-" * 60)
    print(f"Average Win: ${metrics.avg_win:.2f}")
    print(f"Average Loss: ${metrics.avg_loss:.2f}")
    print(f"Largest Win: ${metrics.largest_win:.2f}")
    print(f"Largest Loss: ${metrics.largest_loss:.2f}")
    print("=" * 60)

    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run PolyBot backtests")
    parser.add_argument(
        "--strategy",
        type=str,
        default="arbitrage",
        choices=["arbitrage", "ai_analyst"],
        help="Strategy to backtest",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to backtest",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=10000.0,
        help="Initial balance",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=5.0,
        help="Position size as percent of portfolio",
    )

    args = parser.parse_args()

    setup_logging()

    asyncio.run(
        run_backtest(
            strategy_name=args.strategy,
            days=args.days,
            initial_balance=args.balance,
            position_size_pct=args.position_size,
        )
    )


if __name__ == "__main__":
    main()
