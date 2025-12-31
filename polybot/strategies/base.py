"""
Base strategy class for all trading strategies.

All strategies must inherit from BaseStrategy and implement:
- scan(): Scan for opportunities using Redis cache
- get_scan_interval(): Seconds between scans
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from polybot.cache import RedisCache
from polybot.models import SignalAction, SignalStatus


@dataclass
class Signal:
    """A trading signal."""
    strategy_name: str
    market_id: str
    market_slug: str
    token_id: str
    action: SignalAction
    confidence: float  # 0.0 to 1.0
    expected_edge: float  # Expected profit margin
    entry_price: float
    exit_price: float | None = None
    reasoning: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    status: SignalStatus = SignalStatus.PENDING

    def is_expired(self) -> bool:
        """Check if signal has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "strategy_name": self.strategy_name,
            "market_id": self.market_id,
            "market_slug": self.market_slug,
            "token_id": self.token_id,
            "action": self.action.value,
            "confidence": self.confidence,
            "expected_edge": self.expected_edge,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "reasoning": self.reasoning,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
        }


@dataclass
class StrategyResult:
    """Result from a strategy scan."""
    signals: list[Signal]
    markets_scanned: int
    scan_duration_ms: float
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Subclasses must implement:
    - name: Strategy identifier
    - scan(): Scan for opportunities
    - get_scan_interval(): Seconds between scans
    """

    def __init__(self):
        """Initialize strategy."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.last_scan: datetime | None = None
        self.total_scans: int = 0
        self.total_signals_generated: int = 0
        self._market_cooldowns: dict[str, datetime] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier."""
        pass

    @abstractmethod
    async def scan(self, cache: RedisCache) -> StrategyResult:
        """
        Scan for trading opportunities.

        Args:
            cache: Redis cache with current market prices

        Returns:
            StrategyResult with signals found
        """
        pass

    @abstractmethod
    def get_scan_interval(self) -> int:
        """Seconds between scans."""
        pass

    def create_signal(
        self,
        market_id: str,
        market_slug: str,
        token_id: str,
        action: SignalAction,
        confidence: float,
        expected_edge: float,
        entry_price: float,
        exit_price: float | None = None,
        reasoning: str = "",
        metrics: dict[str, Any] | None = None,
        expires_in_minutes: int = 30,
    ) -> Signal:
        """
        Create a trading signal.

        Args:
            market_id: Condition ID
            market_slug: Market slug
            token_id: Token ID
            action: Buy/Sell action
            confidence: Confidence level (0-1)
            expected_edge: Expected profit margin
            entry_price: Suggested entry price
            exit_price: Target exit price
            reasoning: Explanation for the signal
            metrics: Additional metrics
            expires_in_minutes: Signal expiration time

        Returns:
            Signal object
        """
        return Signal(
            strategy_name=self.name,
            market_id=market_id,
            market_slug=market_slug,
            token_id=token_id,
            action=action,
            confidence=min(1.0, max(0.0, confidence)),
            expected_edge=expected_edge,
            entry_price=entry_price,
            exit_price=exit_price,
            reasoning=reasoning,
            metrics=metrics or {},
            expires_at=datetime.utcnow() + timedelta(minutes=expires_in_minutes),
        )

    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate a signal before emitting.

        Checks:
        - Confidence >= 0.5
        - Expected edge >= 0.5%
        - Market not in cooldown
        """
        # Minimum confidence
        if signal.confidence < 0.5:
            self.logger.debug(f"Signal rejected: low confidence {signal.confidence:.2f}")
            return False

        # Minimum edge
        if signal.expected_edge < 0.005:  # 0.5%
            self.logger.debug(f"Signal rejected: low edge {signal.expected_edge:.4f}")
            return False

        # Check cooldown
        if self._is_in_cooldown(signal.market_id):
            self.logger.debug(f"Signal rejected: market in cooldown {signal.market_slug}")
            return False

        return True

    def set_cooldown(self, market_id: str, minutes: int = 30):
        """Set cooldown for a market."""
        self._market_cooldowns[market_id] = datetime.utcnow() + timedelta(minutes=minutes)

    def _is_in_cooldown(self, market_id: str) -> bool:
        """Check if market is in cooldown."""
        if market_id not in self._market_cooldowns:
            return False

        cooldown_until = self._market_cooldowns[market_id]
        if datetime.utcnow() > cooldown_until:
            del self._market_cooldowns[market_id]
            return False

        return True

    def get_stats(self) -> dict[str, Any]:
        """Get strategy statistics."""
        return {
            "name": self.name,
            "total_scans": self.total_scans,
            "total_signals_generated": self.total_signals_generated,
            "last_scan": self.last_scan.isoformat() if self.last_scan else None,
            "markets_in_cooldown": len(self._market_cooldowns),
        }

    def check_arbitrage(
        self,
        market_id: int,
        prices: dict[str, float],
    ) -> dict | None:
        """
        Check for arbitrage opportunity (for backtesting).

        This is a helper method that can be overridden by arbitrage strategies.
        """
        return None
