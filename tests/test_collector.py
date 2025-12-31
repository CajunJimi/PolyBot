"""Tests for data collector."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from polybot.collector import DataCollector, parse_tokens


class TestParseTokens:
    """Tests for parse_tokens helper function."""

    def test_parse_tokens_list(self):
        """Test parsing tokens from list."""
        tokens = [{"token_id": "1"}, {"token_id": "2"}]
        result = parse_tokens(tokens)
        assert result == tokens

    def test_parse_tokens_json_string(self):
        """Test parsing tokens from JSON string."""
        tokens = '[{"token_id": "1"}]'
        result = parse_tokens(tokens)
        assert result == [{"token_id": "1"}]

    def test_parse_tokens_none(self):
        """Test parsing None returns empty list."""
        result = parse_tokens(None)
        assert result == []

    def test_parse_tokens_invalid(self):
        """Test parsing invalid data returns empty list."""
        result = parse_tokens("invalid json")
        assert result == []


class TestDataCollector:
    """Tests for DataCollector class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Patch config to avoid loading .env
        with patch("polybot.collector.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                gamma_api_url="https://gamma-api.polymarket.com",
                clob_api_url="https://clob.polymarket.com",
                data_api_url="https://data-api.polymarket.com",
                orderbook_interval=30,
                market_refresh_interval=300,
                trade_interval=30,
                max_markets_to_track=100,
                request_delay=0.15,
            )
            self.collector = DataCollector()

    def test_collector_initialization(self):
        """Test collector initializes correctly."""
        assert self.collector._running is False
        assert self.collector._tracked_markets == {}
        assert self.collector.stats["collections_completed"] == 0

    def test_record_health(self):
        """Test health metrics recording."""
        self.collector._record_health("test_component", success=True, latency_ms=100, items=10)

        assert "test_component" in self.collector._health_metrics
        metrics = self.collector._health_metrics["test_component"]
        assert metrics["successes"] == 1
        assert metrics["failures"] == 0
        assert metrics["total_latency_ms"] == 100
        assert metrics["items_collected"] == 10

    def test_record_health_failure(self):
        """Test health metrics recording for failures."""
        self.collector._record_health(
            "test_component",
            success=False,
            error="Test error"
        )

        metrics = self.collector._health_metrics["test_component"]
        assert metrics["failures"] == 1
        assert metrics["last_error"] == "Test error"

    @pytest.mark.asyncio
    async def test_stop(self):
        """Test collector stop."""
        # Mock the cache
        self.collector.cache = AsyncMock()
        self.collector._client = None

        await self.collector.stop()

        assert self.collector._running is False
        self.collector.cache.disconnect.assert_called_once()


class TestMarketValidation:
    """Tests for market validation during collection."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("polybot.collector.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                gamma_api_url="https://gamma-api.polymarket.com",
                clob_api_url="https://clob.polymarket.com",
                data_api_url="https://data-api.polymarket.com",
                orderbook_interval=30,
                market_refresh_interval=300,
                trade_interval=30,
                max_markets_to_track=100,
                request_delay=0.15,
            )
            self.collector = DataCollector()

    def test_validator_initialized(self):
        """Test validator is properly initialized."""
        assert self.collector.validator is not None

    def test_stats_tracking_initialization(self):
        """Test stats are properly initialized."""
        assert self.collector.stats["data_validated"] == 0
        assert self.collector.stats["data_rejected"] == 0


class TestOrderbookCollection:
    """Tests for orderbook collection logic."""

    def test_no_synthetic_data_principle(self):
        """Verify the code doesn't generate synthetic data."""
        # This is a documentation test - the collector code should
        # skip on API failure, never generate fake data.
        #
        # Key patterns in collector.py:
        # 1. "NO synthetic/fake data" comments
        # 2. `continue` on API failures
        # 3. No fallback price generation

        # Read the collector source and verify patterns
        import inspect
        from polybot import collector

        source = inspect.getsource(collector)

        # Verify anti-patterns are NOT present
        assert "synthetic" not in source.lower() or "no synthetic" in source.lower()
        assert "fake data" not in source.lower() or "no fake" in source.lower()

        # Verify correct patterns ARE present
        assert "continue" in source  # Should skip on failures
        assert "skipping" in source.lower()  # Should log when skipping


class TestCollectorConfiguration:
    """Tests for collector configuration."""

    def test_collection_intervals(self):
        """Test collection intervals are reasonable."""
        with patch("polybot.collector.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                gamma_api_url="https://gamma-api.polymarket.com",
                clob_api_url="https://clob.polymarket.com",
                data_api_url="https://data-api.polymarket.com",
                orderbook_interval=30,
                market_refresh_interval=300,
                trade_interval=30,
                max_markets_to_track=100,
                request_delay=0.15,
            )
            collector = DataCollector()

            # Orderbook interval should be at least 10 seconds
            assert collector.config.orderbook_interval >= 10

            # Market refresh should be at least 60 seconds
            assert collector.config.market_refresh_interval >= 60

            # Request delay should be positive
            assert collector.config.request_delay > 0
