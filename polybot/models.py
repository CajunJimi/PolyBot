"""
SQLAlchemy ORM models for PolyBot.

Tables:
- markets: Market metadata
- orderbooks: Time-series orderbook snapshots (TimescaleDB hypertable)
- trades: Trade stream (TimescaleDB hypertable)
- backtest_results: Backtest run results
- signals: Generated trading signals
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class OrderSide(str, Enum):
    """Trade side."""
    BUY = "BUY"
    SELL = "SELL"


class SignalAction(str, Enum):
    """Signal action types."""
    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"
    SELL_YES = "sell_yes"
    SELL_NO = "sell_no"
    BUY_ALL = "buy_all"  # For arbitrage
    SKIP = "skip"


class SignalStatus(str, Enum):
    """Signal status."""
    PENDING = "pending"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class Market(Base):
    """Market metadata from Polymarket."""

    __tablename__ = "markets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    condition_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    slug: Mapped[str] = mapped_column(String(200), nullable=False)
    question: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(100))
    tokens: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    closed: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolution: Mapped[str | None] = mapped_column(String(50))
    end_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    volume_24h: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=0)
    liquidity: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    orderbooks: Mapped[list["OrderBook"]] = relationship(back_populates="market")
    trades: Mapped[list["Trade"]] = relationship(back_populates="market")
    signals: Mapped[list["Signal"]] = relationship(back_populates="market")

    __table_args__ = (
        Index("idx_markets_active_volume", "active", "volume_24h"),
        Index("idx_markets_slug", "slug"),
    )

    def __repr__(self) -> str:
        return f"<Market {self.slug}>"


class OrderBook(Base):
    """Orderbook snapshot."""

    __tablename__ = "orderbooks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    market_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("markets.id", ondelete="CASCADE"), nullable=False
    )
    token_id: Mapped[str] = mapped_column(String(100), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    best_bid: Mapped[float | None] = mapped_column()
    best_ask: Mapped[float | None] = mapped_column()
    mid_price: Mapped[float | None] = mapped_column()
    spread: Mapped[float | None] = mapped_column()
    spread_bps: Mapped[float | None] = mapped_column()
    depth_1pct: Mapped[float | None] = mapped_column()
    bid_depth: Mapped[float | None] = mapped_column()
    ask_depth: Mapped[float | None] = mapped_column()
    imbalance: Mapped[float | None] = mapped_column()
    bids: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    asks: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    outcome: Mapped[str | None] = mapped_column(String(100))

    # Relationships
    market: Mapped["Market"] = relationship(back_populates="orderbooks")

    __table_args__ = (
        Index("idx_orderbooks_market_time", "market_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<OrderBook market_id={self.market_id} timestamp={self.timestamp}>"


class Trade(Base):
    """Trade from Polymarket Data API (TimescaleDB hypertable)."""

    __tablename__ = "trades"

    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    trade_id: Mapped[str] = mapped_column(String(100), primary_key=True, nullable=False)
    market_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("markets.id", ondelete="CASCADE"), nullable=False
    )
    condition_id: Mapped[str | None] = mapped_column(String(100))
    token_id: Mapped[str | None] = mapped_column(String(100))
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    outcome: Mapped[str | None] = mapped_column(String(50))
    price: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    size: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    value_usd: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    maker_address: Mapped[str | None] = mapped_column(String(100))
    taker_address: Mapped[str | None] = mapped_column(String(100))
    is_whale: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    market: Mapped["Market"] = relationship(back_populates="trades")

    __table_args__ = (
        Index("idx_trades_market_time", "market_id", "time"),
    )

    def __repr__(self) -> str:
        return f"<Trade {self.trade_id}>"


class BacktestResult(Base):
    """Backtest run results."""

    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)
    strategy_config: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    initial_balance: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=10000)
    final_balance: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    total_return: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    total_return_pct: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(Numeric(6, 3))
    sortino_ratio: Mapped[Decimal | None] = mapped_column(Numeric(6, 3))
    max_drawdown_pct: Mapped[Decimal | None] = mapped_column(Numeric(6, 3))
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)
    losing_trades: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    profit_factor: Mapped[Decimal | None] = mapped_column(Numeric(6, 3))
    avg_win: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    avg_loss: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    trade_details: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    equity_curve: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    __table_args__ = (
        Index("idx_backtest_strategy", "strategy_name", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<BacktestResult {self.strategy_name} {self.start_date}-{self.end_date}>"


class Signal(Base):
    """Generated trading signal."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)
    market_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("markets.id"), nullable=True
    )
    market_slug: Mapped[str | None] = mapped_column(String(200))
    token_id: Mapped[str | None] = mapped_column(String(100))
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False)
    expected_edge: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    entry_price: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    reasoning: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default=SignalStatus.PENDING.value)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    # Relationships
    market: Mapped["Market"] = relationship(back_populates="signals")

    __table_args__ = (
        Index("idx_signals_status", "status", "created_at"),
        Index("idx_signals_strategy", "strategy_name", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Signal {self.strategy_name} {self.action} {self.market_slug}>"


class CollectionHealth(Base):
    """Collection health metrics."""

    __tablename__ = "collection_health"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    component: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_minutes: Mapped[int] = mapped_column(Integer, default=1)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    successes: Mapped[int] = mapped_column(Integer, default=0)
    failures: Mapped[int] = mapped_column(Integer, default=0)
    avg_latency_ms: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    items_collected: Mapped[int] = mapped_column(Integer, default=0)
    gap_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("idx_health_component_time", "component", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<CollectionHealth {self.component} {self.timestamp}>"
