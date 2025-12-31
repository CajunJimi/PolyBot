"""
Order Execution Engine for Polymarket Trading.

Features:
- Pre-trade validation (price, size, market checks)
- Execution with retry logic and exponential backoff
- Slippage monitoring and protection
- Circuit breaker for failure prevention
- Paper trading mode support
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from sqlalchemy import select

from polybot.db import get_session
from polybot.models import Market, OrderBook, Signal, SignalStatus

logger = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    """Order execution status."""
    PENDING = "pending"
    VALIDATING = "validating"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"
    EXPIRED = "expired"


class OrderType(str, Enum):
    """Order types supported."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class TradingMode(str, Enum):
    """Trading mode."""
    DISABLED = "disabled"
    PAPER = "paper"
    LIVE = "live"
    HALTED = "halted"


@dataclass
class OrderRequest:
    """Order execution request."""
    signal_id: int
    market_id: str  # Condition ID
    token_id: str
    side: str  # "BUY" or "SELL"
    size_usd: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    slippage_tolerance: float = 2.0  # Max slippage % allowed
    timeout_seconds: int = 30
    retry_attempts: int = 3


@dataclass
class OrderResult:
    """Order execution result."""
    order_id: str | None = None
    status: OrderStatus = OrderStatus.PENDING
    executed_price: float | None = None
    executed_size: float | None = None
    executed_value: float | None = None
    fees_paid: float | None = None
    slippage_pct: float | None = None
    execution_time_ms: int | None = None
    error_message: str | None = None
    retry_count: int = 0
    timestamps: dict[str, datetime] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamps:
            self.timestamps = {"created": datetime.utcnow()}


@dataclass
class ExecutionMetrics:
    """Execution performance metrics."""
    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    cancelled_orders: int = 0
    avg_execution_time_ms: float = 0.0
    avg_slippage_pct: float = 0.0
    max_slippage_pct: float = 0.0
    total_fees_paid: float = 0.0
    total_volume_executed: float = 0.0
    retry_rate: float = 0.0
    success_rate: float = 0.0


class OrderExecutor:
    """
    Robust order execution engine.

    Handles all aspects of order execution including validation,
    risk checks, execution, and monitoring.
    """

    def __init__(self, trading_mode: TradingMode = TradingMode.PAPER):
        """Initialize order executor."""
        self.trading_mode = trading_mode
        self.metrics = ExecutionMetrics()
        self.active_orders: dict[str, OrderResult] = {}

        # Configuration
        self.max_slippage_pct = 5.0
        self.price_staleness_seconds = 60
        self.min_order_size_usd = 10.0
        self.max_order_size_usd = 10000.0

        # Circuit breaker
        self.max_failures_per_hour = 10
        self.failure_timestamps: list[datetime] = []

        logger.info(f"OrderExecutor initialized in {trading_mode.value} mode")

    async def execute_signal(self, signal: Signal) -> tuple[bool, OrderResult]:
        """
        Execute a trading signal.

        Args:
            signal: Signal to execute

        Returns:
            Tuple of (success, result)
        """
        try:
            # Check circuit breaker
            if self._is_circuit_breaker_open():
                return False, OrderResult(
                    status=OrderStatus.FAILED,
                    error_message="Circuit breaker open - too many recent failures",
                )

            # Check trading mode
            if self.trading_mode == TradingMode.DISABLED:
                return False, OrderResult(
                    status=OrderStatus.REJECTED,
                    error_message="Trading is disabled",
                )

            if self.trading_mode == TradingMode.HALTED:
                return False, OrderResult(
                    status=OrderStatus.REJECTED,
                    error_message="Trading is halted",
                )

            # Create order request
            order_request = await self._create_order_request(signal)
            if not order_request:
                return False, OrderResult(
                    status=OrderStatus.FAILED,
                    error_message="Failed to create order request",
                )

            # Execute the order
            result = await self._execute_order(order_request)

            # Update metrics
            self._update_metrics(result)

            success = result.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]

            if success:
                logger.info(f"Signal executed successfully: {signal.market_slug}")
            else:
                logger.error(f"Signal execution failed: {result.error_message}")
                self.failure_timestamps.append(datetime.utcnow())

            return success, result

        except Exception as e:
            logger.error(f"Execution error: {e}")
            self.failure_timestamps.append(datetime.utcnow())
            return False, OrderResult(
                status=OrderStatus.FAILED,
                error_message=str(e),
            )

    async def _create_order_request(self, signal: Signal) -> OrderRequest | None:
        """Create order request from signal."""
        try:
            # Determine size (use signal metrics or default)
            size_usd = signal.metrics.get("suggested_size", 50.0) if signal.metrics else 50.0

            # Validate size
            if size_usd < self.min_order_size_usd:
                size_usd = self.min_order_size_usd
            if size_usd > self.max_order_size_usd:
                size_usd = self.max_order_size_usd

            # Determine side from action
            side = "BUY" if "buy" in signal.action.lower() else "SELL"

            return OrderRequest(
                signal_id=signal.id,
                market_id=signal.market_id or "",
                token_id=signal.token_id or "",
                side=side,
                size_usd=size_usd,
                slippage_tolerance=min(self.max_slippage_pct, 2.0),
            )

        except Exception as e:
            logger.error(f"Failed to create order request: {e}")
            return None

    async def _execute_order(self, order_request: OrderRequest) -> OrderResult:
        """Execute order with retry logic."""
        result = OrderResult()
        result.timestamps["validation_start"] = datetime.utcnow()

        # Pre-execution validation
        valid, error = await self._validate_order(order_request)
        if not valid:
            result.status = OrderStatus.REJECTED
            result.error_message = error
            return result

        result.status = OrderStatus.VALIDATING
        result.timestamps["validation_complete"] = datetime.utcnow()

        # Execute with retry logic
        for attempt in range(order_request.retry_attempts + 1):
            try:
                result.retry_count = attempt

                if attempt > 0:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)

                    # Re-validate
                    valid, error = await self._validate_order(order_request)
                    if not valid:
                        result.status = OrderStatus.REJECTED
                        result.error_message = f"Retry validation failed: {error}"
                        break

                # Attempt execution
                success = await self._attempt_execution(order_request, result)
                if success:
                    break

                if attempt < order_request.retry_attempts:
                    logger.warning(f"Execution attempt {attempt + 1} failed, retrying...")

            except Exception as e:
                logger.error(f"Execution attempt {attempt + 1} error: {e}")
                result.error_message = str(e)

                if attempt == order_request.retry_attempts:
                    result.status = OrderStatus.FAILED

        return result

    async def _validate_order(self, order_request: OrderRequest) -> tuple[bool, str | None]:
        """Validate order before execution."""
        try:
            # Validate order size
            if order_request.size_usd < self.min_order_size_usd:
                return False, f"Order size ${order_request.size_usd:.2f} below minimum"

            if order_request.size_usd > self.max_order_size_usd:
                return False, f"Order size ${order_request.size_usd:.2f} above maximum"

            # Validate market exists and is active
            async with get_session() as session:
                result = await session.execute(
                    select(Market).where(Market.condition_id == order_request.market_id)
                )
                market = result.scalar_one_or_none()

                if not market:
                    return False, f"Market {order_request.market_id} not found"

                if not market.active:
                    return False, f"Market is not active"

                if market.closed:
                    return False, f"Market is closed"

            # Get current price
            current_price = await self._get_current_price(order_request.token_id)
            if current_price is None:
                return False, f"Unable to get current price"

            return True, None

        except Exception as e:
            logger.error(f"Order validation error: {e}")
            return False, f"Validation error: {e}"

    async def _attempt_execution(self, order_request: OrderRequest, result: OrderResult) -> bool:
        """Attempt to execute order."""
        try:
            result.status = OrderStatus.SUBMITTING
            result.timestamps["execution_start"] = datetime.utcnow()

            # Get current price
            current_price = await self._get_current_price(order_request.token_id)
            if current_price is None:
                result.error_message = "Unable to get current market price"
                return False

            # Calculate shares
            shares = order_request.size_usd / current_price if current_price > 0 else 0

            if self.trading_mode == TradingMode.PAPER:
                # Paper trading - simulate execution
                success = await self._simulate_execution(order_request, current_price, shares, result)
            else:
                # Live trading - would call Polymarket API
                # For now, simulate
                success = await self._simulate_execution(order_request, current_price, shares, result)

            if success:
                result.status = OrderStatus.FILLED
                result.timestamps["execution_complete"] = datetime.utcnow()

                if "execution_start" in result.timestamps:
                    delta = result.timestamps["execution_complete"] - result.timestamps["execution_start"]
                    result.execution_time_ms = int(delta.total_seconds() * 1000)

                logger.info(f"Order executed: {shares:.2f} shares at ${current_price:.4f}")
                return True

            return False

        except Exception as e:
            logger.error(f"Execution attempt failed: {e}")
            result.error_message = str(e)
            result.status = OrderStatus.FAILED
            return False

    async def _simulate_execution(
        self,
        order_request: OrderRequest,
        price: float,
        shares: float,
        result: OrderResult,
    ) -> bool:
        """Simulate order execution for paper trading."""
        try:
            import random

            # Small delay
            await asyncio.sleep(0.1)

            # Random slippage 0-1%
            slippage_pct = random.uniform(0, 1.0)

            if order_request.side == "BUY":
                executed_price = price * (1 + slippage_pct / 100)
            else:
                executed_price = price * (1 - slippage_pct / 100)

            # Check slippage tolerance
            if slippage_pct > order_request.slippage_tolerance:
                result.error_message = f"Slippage {slippage_pct:.2f}% exceeds tolerance"
                return False

            # Calculate execution details
            executed_value = shares * executed_price
            fees_paid = executed_value * 0.001  # 0.1% fee

            result.executed_price = executed_price
            result.executed_size = shares
            result.executed_value = executed_value
            result.fees_paid = fees_paid
            result.slippage_pct = slippage_pct
            result.order_id = f"paper_{int(time.time())}_{order_request.signal_id}"

            logger.info(
                f"Execution: ${executed_value:.2f} at ${executed_price:.4f} "
                f"(slippage: {slippage_pct:.2f}%)"
            )

            return True

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return False

    async def _get_current_price(self, token_id: str) -> float | None:
        """Get current market price for token."""
        try:
            async with get_session() as session:
                result = await session.execute(
                    select(OrderBook)
                    .where(OrderBook.token_id == token_id)
                    .order_by(OrderBook.time.desc())
                    .limit(1)
                )
                orderbook = result.scalar_one_or_none()

                if not orderbook:
                    return None

                if orderbook.best_bid and orderbook.best_ask:
                    return float((orderbook.best_bid + orderbook.best_ask) / 2)
                elif orderbook.mid_price:
                    return float(orderbook.mid_price)

                return None

        except Exception as e:
            logger.error(f"Failed to get current price: {e}")
            return None

    def _update_metrics(self, result: OrderResult):
        """Update execution metrics."""
        self.metrics.total_orders += 1

        if result.status == OrderStatus.FILLED:
            self.metrics.successful_orders += 1

            if result.executed_value:
                self.metrics.total_volume_executed += result.executed_value

            if result.fees_paid:
                self.metrics.total_fees_paid += result.fees_paid

            if result.execution_time_ms:
                n = self.metrics.successful_orders
                old_avg = self.metrics.avg_execution_time_ms
                self.metrics.avg_execution_time_ms = ((old_avg * (n - 1)) + result.execution_time_ms) / n

            if result.slippage_pct is not None:
                n = self.metrics.successful_orders
                old_avg = self.metrics.avg_slippage_pct
                self.metrics.avg_slippage_pct = ((old_avg * (n - 1)) + result.slippage_pct) / n

                if result.slippage_pct > self.metrics.max_slippage_pct:
                    self.metrics.max_slippage_pct = result.slippage_pct

        elif result.status == OrderStatus.CANCELLED:
            self.metrics.cancelled_orders += 1
        else:
            self.metrics.failed_orders += 1

        if self.metrics.total_orders > 0:
            self.metrics.success_rate = (self.metrics.successful_orders / self.metrics.total_orders) * 100

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker should halt executions."""
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        self.failure_timestamps = [t for t in self.failure_timestamps if t > one_hour_ago]
        return len(self.failure_timestamps) >= self.max_failures_per_hour

    def get_metrics(self) -> dict[str, Any]:
        """Get execution metrics."""
        return {
            "total_orders": self.metrics.total_orders,
            "successful_orders": self.metrics.successful_orders,
            "failed_orders": self.metrics.failed_orders,
            "success_rate_pct": round(self.metrics.success_rate, 2),
            "avg_execution_time_ms": round(self.metrics.avg_execution_time_ms, 2),
            "avg_slippage_pct": round(self.metrics.avg_slippage_pct, 4),
            "total_volume_executed": round(self.metrics.total_volume_executed, 2),
            "circuit_breaker_open": self._is_circuit_breaker_open(),
        }


# Singleton
_executor: OrderExecutor | None = None


def get_order_executor(trading_mode: TradingMode = TradingMode.PAPER) -> OrderExecutor:
    """Get singleton order executor."""
    global _executor
    if _executor is None:
        _executor = OrderExecutor(trading_mode)
    return _executor
