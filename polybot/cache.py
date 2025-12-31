"""
Redis cache for real-time price access.

The collector writes prices to Redis, strategies read from Redis.
This decouples data collection from strategy execution.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import redis.asyncio as redis

from polybot.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class PriceSnapshot:
    """Price snapshot for a single outcome."""
    outcome: str
    token_id: str
    mid_price: float
    best_bid: float
    best_ask: float
    spread: float
    spread_bps: int
    updated_at: datetime


@dataclass
class MarketPrices:
    """Aggregated prices for a market."""
    condition_id: str
    slug: str
    yes_price: float
    no_price: float
    yes_token_id: str = ""
    no_token_id: str = ""
    volume_24h: float = 0.0
    liquidity: float = 0.0
    spread: float = 0.0
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def sum_price(self) -> float:
        """Sum of YES and NO prices."""
        return self.yes_price + self.no_price

    @property
    def arbitrage_gap(self) -> float:
        """Gap from $1.00 (positive = underpriced)."""
        return abs(1.0 - self.sum_price)

    @property
    def is_underpriced(self) -> bool:
        """Market is underpriced (sum < 1)."""
        return self.sum_price < 1.0


class RedisCache:
    """
    Redis-backed price cache for real-time strategy access.

    Key schema:
    - price:{condition_id}:{outcome} -> PriceSnapshot JSON
    - market:{condition_id} -> MarketPrices JSON
    - markets:active -> Set of active condition_ids

    TTL: 5 minutes (stale data cleanup)
    """

    TTL_SECONDS = 300  # 5 minutes

    def __init__(self):
        """Initialize Redis connection."""
        self._client: redis.Redis | None = None
        self._connected = False

    async def connect(self) -> bool:
        """Connect to Redis."""
        if self._connected and self._client:
            return True

        try:
            config = get_config()
            self._client = redis.from_url(
                config.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info("Redis connected")
            return True
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
            self._client = None
            self._connected = False
            logger.info("Redis disconnected")

    async def _ensure_connected(self) -> bool:
        """Ensure Redis is connected."""
        if not self._connected:
            return await self.connect()
        return True

    async def update_price(
        self,
        market_id: str,
        outcome: str,
        mid_price: float,
        token_id: str = "",
        best_bid: float = 0.0,
        best_ask: float = 0.0,
        spread: float = 0.0,
        spread_bps: int = 0,
        volume_24h: float = 0.0,
        liquidity: float = 0.0,
    ) -> bool:
        """
        Update price for a market outcome.

        Args:
            market_id: Condition ID
            outcome: "Yes" or "No"
            mid_price: Mid price
            token_id: Token ID
            best_bid: Best bid price
            best_ask: Best ask price
            spread: Bid-ask spread
            spread_bps: Spread in basis points
            volume_24h: 24h volume
            liquidity: Market liquidity

        Returns:
            True if update succeeded
        """
        if not await self._ensure_connected():
            return False

        try:
            snapshot = {
                "outcome": outcome,
                "token_id": token_id,
                "mid_price": mid_price,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "spread_bps": spread_bps,
                "updated_at": datetime.utcnow().isoformat(),
            }

            # Store individual outcome price
            key = f"price:{market_id}:{outcome.lower()}"
            await self._client.setex(key, self.TTL_SECONDS, json.dumps(snapshot))

            # Update market aggregate
            market_key = f"market:{market_id}"
            existing = await self._client.get(market_key)

            if existing:
                market_data = json.loads(existing)
            else:
                market_data = {
                    "condition_id": market_id,
                    "slug": "",
                    "yes_price": 0.5,
                    "no_price": 0.5,
                    "yes_token_id": "",
                    "no_token_id": "",
                    "volume_24h": 0,
                    "liquidity": 0,
                    "spread": 0,
                }

            # Update the appropriate outcome
            outcome_lower = outcome.lower()
            if outcome_lower == "yes":
                market_data["yes_price"] = mid_price
                market_data["yes_token_id"] = token_id
            elif outcome_lower == "no":
                market_data["no_price"] = mid_price
                market_data["no_token_id"] = token_id

            market_data["volume_24h"] = volume_24h
            market_data["liquidity"] = liquidity
            market_data["spread"] = spread
            market_data["updated_at"] = datetime.utcnow().isoformat()

            await self._client.setex(market_key, self.TTL_SECONDS, json.dumps(market_data))

            # Add to active markets set
            await self._client.sadd("markets:active", market_id)

            return True

        except Exception as e:
            logger.error(f"Failed to update price in cache: {e}")
            return False

    async def set_market_slug(self, market_id: str, slug: str) -> bool:
        """Set market slug in cache."""
        if not await self._ensure_connected():
            return False

        try:
            market_key = f"market:{market_id}"
            existing = await self._client.get(market_key)

            if existing:
                market_data = json.loads(existing)
                market_data["slug"] = slug
                await self._client.setex(market_key, self.TTL_SECONDS, json.dumps(market_data))
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to set market slug: {e}")
            return False

    async def get_market_prices(self, market_id: str) -> MarketPrices | None:
        """Get aggregated prices for a market."""
        if not await self._ensure_connected():
            return None

        try:
            market_key = f"market:{market_id}"
            data = await self._client.get(market_key)

            if not data:
                return None

            market_data = json.loads(data)
            return MarketPrices(
                condition_id=market_data.get("condition_id", market_id),
                slug=market_data.get("slug", ""),
                yes_price=float(market_data.get("yes_price", 0.5)),
                no_price=float(market_data.get("no_price", 0.5)),
                yes_token_id=market_data.get("yes_token_id", ""),
                no_token_id=market_data.get("no_token_id", ""),
                volume_24h=float(market_data.get("volume_24h", 0)),
                liquidity=float(market_data.get("liquidity", 0)),
                spread=float(market_data.get("spread", 0)),
                updated_at=datetime.fromisoformat(market_data["updated_at"])
                if market_data.get("updated_at")
                else datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Failed to get market prices: {e}")
            return None

    async def get_all_markets(self) -> list[MarketPrices]:
        """Get all cached markets."""
        if not await self._ensure_connected():
            return []

        try:
            # Get active market IDs
            market_ids = await self._client.smembers("markets:active")

            markets = []
            for market_id in market_ids:
                prices = await self.get_market_prices(market_id)
                if prices:
                    markets.append(prices)

            return markets

        except Exception as e:
            logger.error(f"Failed to get all markets: {e}")
            return []

    async def get_markets_with_arbitrage(
        self,
        min_gap: float = 0.005,
        max_gap: float = 0.15,
        min_liquidity: float = 1000,
    ) -> list[MarketPrices]:
        """
        Get markets with potential arbitrage opportunities.

        Args:
            min_gap: Minimum gap from $1.00 (default 0.5%)
            max_gap: Maximum gap (filter suspicious data)
            min_liquidity: Minimum liquidity requirement

        Returns:
            List of MarketPrices with arbitrage potential
        """
        markets = await self.get_all_markets()

        return [
            m for m in markets
            if min_gap <= m.arbitrage_gap <= max_gap
            and m.liquidity >= min_liquidity
        ]

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if not await self._ensure_connected():
            return {"connected": False}

        try:
            active_count = await self._client.scard("markets:active")
            info = await self._client.info("memory")

            return {
                "connected": True,
                "active_markets": active_count,
                "memory_used": info.get("used_memory_human", "unknown"),
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"connected": False, "error": str(e)}

    async def clear_all(self) -> bool:
        """Clear all cached data."""
        if not await self._ensure_connected():
            return False

        try:
            await self._client.flushdb()
            logger.info("Cache cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False


# Singleton instance
_cache: RedisCache | None = None


def get_redis_cache() -> RedisCache:
    """Get global Redis cache instance."""
    global _cache
    if _cache is None:
        _cache = RedisCache()
    return _cache
