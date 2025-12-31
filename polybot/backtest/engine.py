"""
Backtest engine for strategy validation.

Runs strategies against historical data to calculate performance metrics.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from polybot.db import get_session
from polybot.models import Market, OrderBook, BacktestResult
from polybot.backtest.metrics import calculate_metrics, BacktestMetrics

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """A trading position."""
    market_id: int
    market_slug: str
    outcome: str
    entry_price: float
    size: float
    entry_time: datetime
    exit_price: float | None = None
    exit_time: datetime | None = None
    pnl: float = 0.0
    return_pct: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def duration_hours(self) -> float:
        if self.exit_time and self.entry_time:
            return (self.exit_time - self.entry_time).total_seconds() / 3600
        return 0.0

    def close(self, exit_price: float, exit_time: datetime):
        """Close the position and calculate P&L."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.pnl = (exit_price - self.entry_price) * self.size
        self.return_pct = ((exit_price - self.entry_price) / self.entry_price * 100) if self.entry_price > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_slug": self.market_slug,
            "outcome": self.outcome,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "pnl": self.pnl,
            "return_pct": self.return_pct,
            "duration_hours": self.duration_hours,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
        }


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_balance: float = 10000.0
    position_size_pct: float = 5.0  # % of portfolio per trade
    max_positions: int = 10
    slippage_pct: float = 0.1  # 0.1% slippage
    commission_pct: float = 0.0  # No commission on Polymarket
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None


@dataclass
class BacktestState:
    """Current state during backtest execution."""
    balance: float
    positions: list[Position] = field(default_factory=list)
    closed_trades: list[Position] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.is_open]

    @property
    def total_equity(self) -> float:
        # Balance + unrealized P&L (approximated)
        return self.balance + sum(p.pnl for p in self.open_positions)


class BacktestEngine:
    """
    Engine for running strategy backtests.

    Usage:
        engine = BacktestEngine()
        result = await engine.run(config, strategy)
    """

    def __init__(self):
        """Initialize backtest engine."""
        self.logger = logging.getLogger(f"{__name__}.BacktestEngine")

    async def run(
        self,
        config: BacktestConfig,
        strategy: "BaseStrategy",
    ) -> BacktestMetrics:
        """
        Run a backtest for a strategy.

        Args:
            config: Backtest configuration
            strategy: Strategy instance to test

        Returns:
            BacktestMetrics with performance results
        """
        self.logger.info(
            f"Starting backtest: {config.strategy_name} "
            f"from {config.start_date} to {config.end_date}"
        )

        state = BacktestState(
            balance=config.initial_balance,
            equity_curve=[config.initial_balance],
        )

        async with get_session() as session:
            # Get all markets with historical data in date range
            markets = await self._get_markets_with_data(
                session, config.start_date, config.end_date
            )

            self.logger.info(f"Found {len(markets)} markets with historical data")

            # Process day by day
            current_date = config.start_date
            while current_date <= config.end_date:
                # Get price data for this day
                prices = await self._get_prices_for_date(session, current_date, markets)

                if prices:
                    # Run strategy scan
                    signals = await self._run_strategy_scan(strategy, prices)

                    # Process signals
                    for signal in signals:
                        await self._process_signal(
                            state, config, signal, prices, current_date
                        )

                    # Check stop loss / take profit
                    await self._check_exits(state, config, prices, current_date)

                    # Record equity
                    state.equity_curve.append(state.total_equity)

                current_date += timedelta(days=1)

            # Close any remaining positions at end
            final_prices = await self._get_prices_for_date(
                session, config.end_date, markets
            )
            for position in state.open_positions:
                market_prices = final_prices.get(position.market_id, {})
                exit_price = market_prices.get(position.outcome, position.entry_price)
                position.close(exit_price, config.end_date)
                state.closed_trades.append(position)
                state.balance += position.pnl

        # Calculate metrics
        trades = [p.to_dict() for p in state.closed_trades]
        metrics = calculate_metrics(
            trades=trades,
            equity_curve=state.equity_curve,
            initial_balance=config.initial_balance,
        )

        # Save results to database
        await self._save_results(config, metrics, trades, state.equity_curve)

        self.logger.info(
            f"Backtest complete: {len(trades)} trades, "
            f"return: {metrics.total_return_pct:.2f}%, "
            f"Sharpe: {metrics.sharpe_ratio:.2f}"
        )

        return metrics

    async def _get_markets_with_data(
        self,
        session: AsyncSession,
        start_date: datetime,
        end_date: datetime,
    ) -> list[Market]:
        """Get markets that have historical data in date range."""
        result = await session.execute(
            select(Market)
            .where(Market.active == True)
            .order_by(Market.volume_24h.desc())
            .limit(100)
        )
        return list(result.scalars().all())

    async def _get_prices_for_date(
        self,
        session: AsyncSession,
        date: datetime,
        markets: list[Market],
    ) -> dict[int, dict[str, float]]:
        """Get prices for all markets on a specific date."""
        prices = {}

        for market in markets:
            # Get latest orderbook for this date
            result = await session.execute(
                select(OrderBook)
                .where(
                    OrderBook.market_id == market.id,
                    OrderBook.time >= date,
                    OrderBook.time < date + timedelta(days=1),
                )
                .order_by(OrderBook.time.desc())
                .limit(2)  # YES and NO
            )
            orderbooks = list(result.scalars().all())

            if orderbooks:
                market_prices = {}
                for ob in orderbooks:
                    if ob.outcome and ob.mid_price:
                        market_prices[ob.outcome] = float(ob.mid_price)
                if market_prices:
                    prices[market.id] = market_prices

        return prices

    async def _run_strategy_scan(
        self,
        strategy: "BaseStrategy",
        prices: dict[int, dict[str, float]],
    ) -> list[dict]:
        """Run strategy scan with current prices."""
        # Create a mock cache/data store for the strategy
        # This is simplified - in production you'd have proper data access
        try:
            # The strategy.scan() method expects a cache interface
            # For backtesting, we provide prices directly
            signals = []

            # Simplified: check for arbitrage opportunities
            if hasattr(strategy, "check_arbitrage"):
                for market_id, market_prices in prices.items():
                    signal = strategy.check_arbitrage(market_id, market_prices)
                    if signal:
                        signals.append(signal)

            return signals
        except Exception as e:
            self.logger.warning(f"Strategy scan error: {e}")
            return []

    async def _process_signal(
        self,
        state: BacktestState,
        config: BacktestConfig,
        signal: dict,
        prices: dict[int, dict[str, float]],
        current_date: datetime,
    ):
        """Process a trading signal."""
        if len(state.open_positions) >= config.max_positions:
            return  # Max positions reached

        market_id = signal.get("market_id")
        outcome = signal.get("outcome", "Yes")
        confidence = signal.get("confidence", 0)

        if confidence < 0.6:  # Minimum confidence threshold
            return

        # Get entry price
        market_prices = prices.get(market_id, {})
        entry_price = market_prices.get(outcome)

        if not entry_price:
            return

        # Apply slippage
        entry_price *= (1 + config.slippage_pct / 100)

        # Calculate position size
        position_value = state.balance * (config.position_size_pct / 100)
        size = position_value / entry_price if entry_price > 0 else 0

        if size <= 0:
            return

        # Open position
        position = Position(
            market_id=market_id,
            market_slug=signal.get("market_slug", f"market_{market_id}"),
            outcome=outcome,
            entry_price=entry_price,
            size=size,
            entry_time=current_date,
        )

        state.positions.append(position)
        state.balance -= position_value

        self.logger.debug(
            f"Opened position: {position.market_slug} {outcome} @ {entry_price:.4f}"
        )

    async def _check_exits(
        self,
        state: BacktestState,
        config: BacktestConfig,
        prices: dict[int, dict[str, float]],
        current_date: datetime,
    ):
        """Check and execute stop loss / take profit exits."""
        for position in state.open_positions:
            market_prices = prices.get(position.market_id, {})
            current_price = market_prices.get(position.outcome)

            if not current_price:
                continue

            should_exit = False

            # Check stop loss
            if config.stop_loss_pct:
                loss_pct = (position.entry_price - current_price) / position.entry_price * 100
                if loss_pct >= config.stop_loss_pct:
                    should_exit = True
                    self.logger.debug(f"Stop loss triggered: {position.market_slug}")

            # Check take profit
            if config.take_profit_pct:
                profit_pct = (current_price - position.entry_price) / position.entry_price * 100
                if profit_pct >= config.take_profit_pct:
                    should_exit = True
                    self.logger.debug(f"Take profit triggered: {position.market_slug}")

            if should_exit:
                # Apply slippage on exit
                exit_price = current_price * (1 - config.slippage_pct / 100)
                position.close(exit_price, current_date)
                state.closed_trades.append(position)
                state.balance += position.size * exit_price

    async def _save_results(
        self,
        config: BacktestConfig,
        metrics: BacktestMetrics,
        trades: list[dict],
        equity_curve: list[float],
    ):
        """Save backtest results to database."""
        async with get_session() as session:
            result = BacktestResult(
                strategy_name=config.strategy_name,
                strategy_config={
                    "position_size_pct": config.position_size_pct,
                    "max_positions": config.max_positions,
                    "slippage_pct": config.slippage_pct,
                    "stop_loss_pct": config.stop_loss_pct,
                    "take_profit_pct": config.take_profit_pct,
                },
                start_date=config.start_date,
                end_date=config.end_date,
                initial_balance=metrics.initial_balance,
                final_balance=metrics.final_balance,
                total_return=metrics.total_return,
                total_return_pct=metrics.total_return_pct,
                sharpe_ratio=metrics.sharpe_ratio,
                sortino_ratio=metrics.sortino_ratio,
                max_drawdown_pct=metrics.max_drawdown_pct,
                total_trades=metrics.total_trades,
                winning_trades=metrics.winning_trades,
                losing_trades=metrics.losing_trades,
                win_rate=metrics.win_rate,
                profit_factor=metrics.profit_factor,
                avg_win=metrics.avg_win,
                avg_loss=metrics.avg_loss,
                trade_details=trades,
                equity_curve=equity_curve,
            )
            session.add(result)

        self.logger.info(f"Backtest results saved: {config.strategy_name}")
