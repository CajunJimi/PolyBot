"""Tests for data validator."""

import pytest

from polybot.validator import DataValidator, ValidationSeverity


class TestDataValidator:
    """Tests for DataValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator()

    def test_validate_market_valid(self):
        """Test validation of valid market data."""
        data = {
            "conditionId": "0x123",
            "slug": "test-market",
            "question": "Will this happen?",
            "volume24hr": 10000,
            "liquidity": 5000,
        }

        result = self.validator.validate_market(data)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_market_missing_condition_id(self):
        """Test validation fails without condition_id."""
        data = {
            "slug": "test-market",
        }

        result = self.validator.validate_market(data)

        assert not result.is_valid
        assert any(e.field == "condition_id" for e in result.errors)

    def test_validate_market_missing_slug(self):
        """Test validation fails without slug."""
        data = {
            "conditionId": "0x123",
        }

        result = self.validator.validate_market(data)

        assert not result.is_valid
        assert any(e.field == "slug" for e in result.errors)

    def test_validate_market_negative_volume(self):
        """Test validation fails with negative volume."""
        data = {
            "conditionId": "0x123",
            "slug": "test-market",
            "volume24hr": -100,
        }

        result = self.validator.validate_market(data)

        assert not result.is_valid
        assert any(e.field == "volume_24h" for e in result.errors)

    def test_validate_orderbook_valid(self):
        """Test validation of valid orderbook data."""
        data = {
            "best_bid": 0.45,
            "best_ask": 0.55,
            "mid_price": 0.50,
            "spread": 0.10,
        }

        result = self.validator.validate_orderbook(data)

        assert result.is_valid

    def test_validate_orderbook_crossed_book(self):
        """Test validation fails with crossed book (bid > ask)."""
        data = {
            "best_bid": 0.60,
            "best_ask": 0.50,
        }

        result = self.validator.validate_orderbook(data)

        assert not result.is_valid
        assert any("Crossed" in e.message for e in result.errors)

    def test_validate_orderbook_price_out_of_range(self):
        """Test validation fails with prices outside 0-1 range."""
        data = {
            "best_bid": 1.5,
            "best_ask": 2.0,
        }

        result = self.validator.validate_orderbook(data)

        assert not result.is_valid
        assert any("out of range" in e.message for e in result.errors)

    def test_validate_orderbook_large_spread_warning(self):
        """Test validation warns on large spread."""
        data = {
            "best_bid": 0.10,
            "best_ask": 0.90,
        }

        result = self.validator.validate_orderbook(data)

        # Should be valid but with warnings
        assert result.is_valid
        assert result.warnings > 0

    def test_validate_trade_valid(self):
        """Test validation of valid trade data."""
        data = {
            "transactionHash": "0xabc123",
            "price": 0.50,
            "size": 100,
            "side": "BUY",
        }

        result = self.validator.validate_trade(data)

        assert result.is_valid

    def test_validate_trade_missing_trade_id(self):
        """Test validation fails without trade_id."""
        data = {
            "price": 0.50,
            "size": 100,
        }

        result = self.validator.validate_trade(data)

        assert not result.is_valid
        assert any(e.field == "trade_id" for e in result.errors)

    def test_validate_trade_negative_size(self):
        """Test validation fails with negative size."""
        data = {
            "transactionHash": "0xabc123",
            "price": 0.50,
            "size": -100,
        }

        result = self.validator.validate_trade(data)

        assert not result.is_valid
        assert any(e.field == "size" for e in result.errors)

    def test_validate_trade_price_out_of_range(self):
        """Test validation fails with price outside 0-1."""
        data = {
            "transactionHash": "0xabc123",
            "price": 1.5,
            "size": 100,
        }

        result = self.validator.validate_trade(data)

        assert not result.is_valid
        assert any("out of range" in e.message for e in result.errors)

    def test_stats_tracking(self):
        """Test that validation stats are tracked."""
        self.validator.reset_stats()

        # Validate some data
        self.validator.validate_market({"conditionId": "0x1", "slug": "test"})
        self.validator.validate_market({})  # Invalid
        self.validator.validate_orderbook({"best_bid": 0.4, "best_ask": 0.6})

        stats = self.validator.get_stats()

        assert stats["total_validated"] == 3
        assert stats["markets_validated"] == 2
        assert stats["markets_rejected"] == 1
        assert stats["orderbooks_validated"] == 1
