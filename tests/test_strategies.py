"""Tests for trading strategies."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from polybot.strategies import ArbitrageStrategy
from polybot.cache import MarketPrices
from polybot.models import SignalAction


class TestArbitrageStrategy:
    """Tests for ArbitrageStrategy class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ArbitrageStrategy(
            min_gap=0.005,  # 0.5%
            max_gap=0.15,   # 15%
            min_liquidity=1000,
        )

    def test_strategy_name(self):
        """Test strategy name."""
        assert self.strategy.name == "arbitrage"

    def test_scan_interval(self):
        """Test scan interval."""
        assert self.strategy.get_scan_interval() == 30

    def test_analyze_market_underpriced(self):
        """Test analyzing underpriced market."""
        prices = {"Yes": 0.45, "No": 0.50}  # Sum = 0.95 (5% gap)

        result = self.strategy.analyze_market(prices)

        assert result["has_opportunity"]
        assert result["type"] == "underpriced"
        assert result["price_sum"] == 0.95
        assert result["gap"] == 0.05

    def test_analyze_market_overpriced(self):
        """Test analyzing overpriced market."""
        prices = {"Yes": 0.55, "No": 0.50}  # Sum = 1.05 (5% gap)

        result = self.strategy.analyze_market(prices)

        assert result["has_opportunity"]
        assert result["type"] == "overpriced"
        assert result["price_sum"] == 1.05

    def test_analyze_market_no_opportunity(self):
        """Test analyzing market with small gap."""
        prices = {"Yes": 0.50, "No": 0.499}  # Sum = 0.999 (0.1% gap)

        result = self.strategy.analyze_market(prices)

        assert not result["has_opportunity"]

    def test_analyze_market_empty_prices(self):
        """Test analyzing market with empty prices."""
        result = self.strategy.analyze_market({})

        assert not result["has_opportunity"]
        assert result["reason"] == "No prices"

    def test_check_arbitrage_opportunity(self):
        """Test check_arbitrage for backtesting."""
        prices = {"Yes": 0.45, "No": 0.50}

        result = self.strategy.check_arbitrage(1, prices)

        assert result is not None
        assert result["action"] == "BUY_ALL"
        assert result["gap"] == 0.05

    def test_check_arbitrage_no_opportunity(self):
        """Test check_arbitrage with no opportunity."""
        prices = {"Yes": 0.50, "No": 0.50}

        result = self.strategy.check_arbitrage(1, prices)

        assert result is None

    def test_check_arbitrage_suspicious_gap(self):
        """Test check_arbitrage filters suspicious large gaps."""
        prices = {"Yes": 0.30, "No": 0.30}  # 40% gap - too large

        result = self.strategy.check_arbitrage(1, prices)

        assert result is None

    @pytest.mark.asyncio
    async def test_scan_empty_cache(self):
        """Test scanning with empty cache."""
        mock_cache = AsyncMock()
        mock_cache.get_all_markets = AsyncMock(return_value=[])

        result = await self.strategy.scan(mock_cache)

        assert result.markets_scanned == 0
        assert len(result.signals) == 0

    @pytest.mark.asyncio
    async def test_scan_with_opportunity(self):
        """Test scanning finds arbitrage opportunity."""
        # Create mock market with arbitrage opportunity
        mock_market = MagicMock(spec=MarketPrices)
        mock_market.condition_id = "0x123"
        mock_market.slug = "test-market"
        mock_market.yes_price = 0.45
        mock_market.no_price = 0.50
        mock_market.yes_token_id = "token1"
        mock_market.no_token_id = "token2"
        mock_market.sum_price = 0.95
        mock_market.arbitrage_gap = 0.05
        mock_market.is_underpriced = True
        mock_market.liquidity = 10000
        mock_market.spread = 0.01

        mock_cache = AsyncMock()
        mock_cache.get_all_markets = AsyncMock(return_value=[mock_market])

        result = await self.strategy.scan(mock_cache)

        assert result.markets_scanned == 1
        assert len(result.signals) == 1
        assert result.signals[0].action == SignalAction.BUY_ALL
        assert result.signals[0].expected_edge == 0.05

    @pytest.mark.asyncio
    async def test_scan_filters_low_liquidity(self):
        """Test scanning filters low liquidity markets."""
        mock_market = MagicMock(spec=MarketPrices)
        mock_market.condition_id = "0x123"
        mock_market.slug = "test-market"
        mock_market.arbitrage_gap = 0.05
        mock_market.liquidity = 100  # Below minimum
        mock_market.is_underpriced = True

        mock_cache = AsyncMock()
        mock_cache.get_all_markets = AsyncMock(return_value=[mock_market])

        result = await self.strategy.scan(mock_cache)

        assert result.markets_scanned == 1
        assert len(result.signals) == 0  # Filtered due to low liquidity


class TestSignalValidation:
    """Tests for signal validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ArbitrageStrategy()

    def test_validate_signal_low_confidence(self):
        """Test signal validation rejects low confidence."""
        signal = self.strategy.create_signal(
            market_id="0x123",
            market_slug="test",
            token_id="token1",
            action=SignalAction.BUY_YES,
            confidence=0.3,  # Below threshold
            expected_edge=0.05,
            entry_price=0.50,
        )

        assert not self.strategy.validate_signal(signal)

    def test_validate_signal_low_edge(self):
        """Test signal validation rejects low edge."""
        signal = self.strategy.create_signal(
            market_id="0x123",
            market_slug="test",
            token_id="token1",
            action=SignalAction.BUY_YES,
            confidence=0.8,
            expected_edge=0.001,  # Below threshold
            entry_price=0.50,
        )

        assert not self.strategy.validate_signal(signal)

    def test_validate_signal_valid(self):
        """Test signal validation accepts valid signal."""
        signal = self.strategy.create_signal(
            market_id="0x123",
            market_slug="test",
            token_id="token1",
            action=SignalAction.BUY_YES,
            confidence=0.8,
            expected_edge=0.05,
            entry_price=0.50,
        )

        assert self.strategy.validate_signal(signal)

    def test_cooldown_mechanism(self):
        """Test market cooldown prevents duplicate signals."""
        signal = self.strategy.create_signal(
            market_id="0x123",
            market_slug="test",
            token_id="token1",
            action=SignalAction.BUY_YES,
            confidence=0.8,
            expected_edge=0.05,
            entry_price=0.50,
        )

        # First signal should be valid
        assert self.strategy.validate_signal(signal)

        # Set cooldown
        self.strategy.set_cooldown("0x123", minutes=30)

        # Second signal should be rejected due to cooldown
        assert not self.strategy.validate_signal(signal)
