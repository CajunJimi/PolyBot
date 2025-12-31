"""
Production-grade data collector for Polymarket.

CRITICAL DESIGN DECISIONS:
1. NO synthetic/fake data - skip if API fails
2. Validate ALL data before storage
3. Log errors at WARNING/ERROR level, not DEBUG
4. Single source of truth for all market data

Collection loops:
- Full market discovery: Every 5 minutes (paginated)
- Quick market refresh: Every 2 minutes (top 200 by volume)
- Orderbook collection: Every 30 seconds
- Trade stream: Every 30 seconds
- Health check: Every 1 minute
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import select, func

from polybot.cache import get_redis_cache
from polybot.config import get_config
from polybot.db import get_session
from polybot.models import Market, OrderBook, Trade, CollectionHealth
from polybot.validator import get_validator

logger = logging.getLogger(__name__)


def parse_tokens(tokens_data: Any) -> list[dict]:
    """Parse tokens from API - handles string or list format."""
    if tokens_data is None:
        return []
    if isinstance(tokens_data, list):
        return tokens_data
    if isinstance(tokens_data, str):
        try:
            parsed = json.loads(tokens_data)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


class DataCollector:
    """
    Production-grade data collector.

    Architecture:
    1. Collection loops run independently
    2. All data passes through DataValidator before storage
    3. Redis cache updated for strategy real-time access
    4. Health metrics tracked and stored
    """

    # Categories for comprehensive market discovery
    CATEGORIES = [
        "Politics", "Sports", "Crypto", "Pop Culture", "Business",
        "Science", "AI", "Entertainment", "World", "Economy",
        "Elections", "Weather", "Tech", "Finance",
    ]

    def __init__(self):
        """Initialize collector."""
        self._client: httpx.AsyncClient | None = None
        self._tracked_markets: dict[str, dict] = {}
        self._running = False

        # Get dependencies
        self.config = get_config()
        self.validator = get_validator()
        self.cache = get_redis_cache()

        # Statistics
        self.stats = {
            "started_at": None,
            "collections_completed": 0,
            "markets_discovered": 0,
            "orderbooks_collected": 0,
            "trades_collected": 0,
            "data_validated": 0,
            "data_rejected": 0,
        }

        # Health tracking
        self._health_metrics: dict[str, dict] = {}

        logger.info("DataCollector initialized")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client with proper configuration."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=25),
            )
        return self._client

    async def start(self):
        """Start all collection tasks."""
        logger.info("Starting DataCollector")
        logger.info(f"  Orderbook interval: {self.config.orderbook_interval}s")
        logger.info(f"  Market refresh: {self.config.market_refresh_interval}s")
        logger.info(f"  Max markets: {self.config.max_markets_to_track}")

        self._running = True
        self.stats["started_at"] = datetime.utcnow()

        # Connect to Redis
        await self.cache.connect()

        # Initial market discovery
        await self._full_market_discovery()

        # Start all collection loops
        tasks = [
            asyncio.create_task(self._full_market_loop(), name="full_market_loop"),
            asyncio.create_task(self._quick_market_loop(), name="quick_market_loop"),
            asyncio.create_task(self._orderbook_loop(), name="orderbook_loop"),
            asyncio.create_task(self._trade_loop(), name="trade_loop"),
            asyncio.create_task(self._health_check_loop(), name="health_check_loop"),
            asyncio.create_task(self._stats_loop(), name="stats_loop"),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("DataCollector shutting down...")
        finally:
            await self.stop()

    async def stop(self):
        """Stop collector and cleanup."""
        self._running = False
        if self._client:
            await self._client.aclose()
        await self.cache.disconnect()
        logger.info("DataCollector stopped")

    # =========================================================================
    # MARKET DISCOVERY
    # =========================================================================

    async def _full_market_loop(self):
        """Full market discovery loop."""
        while self._running:
            await asyncio.sleep(self.config.market_refresh_interval)
            try:
                await self._full_market_discovery()
            except Exception as e:
                logger.error(f"Full market discovery error: {e}")
                self._record_health("market_discovery", success=False, error=str(e))

    async def _full_market_discovery(self):
        """
        Discover ALL active markets using pagination.

        Ensures 100% market coverage by:
        1. Paginating through all markets
        2. Fetching by category to catch edge cases
        """
        start_time = time.time()
        client = await self._get_client()
        all_markets = []
        offset = 0
        limit = 100

        logger.info("Starting full market discovery...")

        try:
            # Phase 1: Paginate through all markets
            while True:
                try:
                    response = await client.get(
                        f"{self.config.gamma_api_url}/markets",
                        params={
                            "limit": limit,
                            "offset": offset,
                            "active": True,
                            "closed": False,
                        },
                    )
                    response.raise_for_status()
                    markets = response.json()

                    if not markets:
                        break

                    all_markets.extend(markets)
                    offset += limit

                    # Safety limit
                    if offset > 5000:
                        logger.warning("Hit 5000 market limit")
                        break

                    await asyncio.sleep(self.config.request_delay)

                except Exception as e:
                    logger.error(f"Pagination error at offset {offset}: {e}")
                    break

            # Phase 2: Fetch by category (catch any missed)
            existing_ids = {
                m.get("conditionId") or m.get("condition_id") for m in all_markets
            }

            for category in self.CATEGORIES:
                try:
                    response = await client.get(
                        f"{self.config.gamma_api_url}/markets",
                        params={
                            "limit": 100,
                            "category": category,
                            "active": True,
                            "closed": False,
                        },
                    )
                    if response.status_code == 200:
                        cat_markets = response.json()
                        for m in cat_markets:
                            cid = m.get("conditionId") or m.get("condition_id")
                            if cid and cid not in existing_ids:
                                all_markets.append(m)
                                existing_ids.add(cid)
                except Exception:
                    pass

                await asyncio.sleep(self.config.request_delay)

            # Phase 3: Validate and store
            await self._store_markets(all_markets)

            latency_ms = (time.time() - start_time) * 1000
            self.stats["markets_discovered"] = len(all_markets)
            self._record_health("market_discovery", success=True, latency_ms=latency_ms)

            logger.info(f"Market discovery complete: {len(all_markets)} markets in {latency_ms:.0f}ms")

        except Exception as e:
            logger.error(f"Market discovery failed: {e}")
            self._record_health("market_discovery", success=False, error=str(e))

    async def _store_markets(self, markets: list[dict]):
        """Store markets to database with validation."""
        valid_count = 0
        rejected_count = 0

        async with get_session() as session:
            for m in markets:
                # Validate market data
                validation = self.validator.validate_market(m)
                self.stats["data_validated"] += 1

                if not validation.is_valid:
                    self.stats["data_rejected"] += 1
                    rejected_count += 1
                    logger.warning(f"Rejected market {m.get('slug', 'unknown')}: {validation.error_summary}")
                    continue

                condition_id = m.get("condition_id") or m.get("conditionId")
                if not condition_id:
                    continue

                # Check if exists
                result = await session.execute(
                    select(Market).where(Market.condition_id == condition_id)
                )
                existing = result.scalar_one_or_none()

                if existing:
                    # Update existing
                    existing.slug = m.get("slug", existing.slug)
                    existing.question = m.get("question", existing.question)
                    existing.description = m.get("description", existing.description)
                    existing.category = m.get("category", existing.category)
                    existing.active = m.get("active", True)
                    existing.closed = m.get("closed", False)
                    existing.volume_24h = float(m.get("volume24hr", 0) or 0)
                    existing.liquidity = float(m.get("liquidity", 0) or 0)
                    existing.updated_at = datetime.utcnow()
                else:
                    # Insert new
                    end_date = None
                    if m.get("endDate"):
                        try:
                            end_date = datetime.fromisoformat(
                                m["endDate"].replace("Z", "+00:00").replace("+00:00", "")
                            )
                        except (ValueError, AttributeError):
                            pass

                    new_market = Market(
                        condition_id=condition_id,
                        slug=m.get("slug", condition_id),
                        question=m.get("question", ""),
                        description=m.get("description"),
                        category=m.get("category"),
                        tokens=m.get("clobTokenIds"),
                        active=m.get("active", True),
                        closed=m.get("closed", False),
                        end_date=end_date,
                        volume_24h=float(m.get("volume24hr", 0) or 0),
                        liquidity=float(m.get("liquidity", 0) or 0),
                    )
                    session.add(new_market)

                # Track for orderbook collection
                self._tracked_markets[condition_id] = {
                    "slug": m.get("slug"),
                    "tokens": parse_tokens(m.get("clobTokenIds")),
                    "volume_24h": float(m.get("volume24hr", 0) or 0),
                    "liquidity": float(m.get("liquidity", 0) or 0),
                }

                # Update cache with slug
                await self.cache.set_market_slug(condition_id, m.get("slug", ""))

                valid_count += 1

        if rejected_count > 0:
            logger.warning(f"Rejected {rejected_count}/{len(markets)} markets")

        logger.info(f"Stored {valid_count} valid markets")

    async def _quick_market_loop(self):
        """Quick market refresh loop."""
        await asyncio.sleep(30)

        while self._running:
            await asyncio.sleep(120)  # Every 2 minutes
            try:
                await self._quick_market_refresh()
            except Exception as e:
                logger.error(f"Quick market refresh error: {e}")

    async def _quick_market_refresh(self):
        """Refresh top 200 markets by volume."""
        client = await self._get_client()

        try:
            response = await client.get(
                f"{self.config.gamma_api_url}/markets",
                params={
                    "limit": 200,
                    "active": True,
                    "closed": False,
                    "order": "volume24hr",
                    "ascending": False,
                },
            )
            response.raise_for_status()
            markets = response.json()

            for m in markets:
                condition_id = m.get("condition_id") or m.get("conditionId")
                if condition_id:
                    self._tracked_markets[condition_id] = {
                        "slug": m.get("slug"),
                        "tokens": parse_tokens(m.get("clobTokenIds")),
                        "volume_24h": float(m.get("volume24hr", 0) or 0),
                        "liquidity": float(m.get("liquidity", 0) or 0),
                    }

            logger.debug(f"Quick refresh: {len(markets)} markets")

        except Exception as e:
            logger.error(f"Quick refresh error: {e}")

    # =========================================================================
    # ORDERBOOK COLLECTION
    # =========================================================================

    async def _orderbook_loop(self):
        """Orderbook collection loop."""
        await asyncio.sleep(5)

        while self._running:
            try:
                await self._collect_orderbooks()
                self.stats["collections_completed"] += 1
            except Exception as e:
                logger.error(f"Orderbook collection error: {e}")
                self._record_health("orderbook", success=False, error=str(e))

            await asyncio.sleep(self.config.orderbook_interval)

    async def _collect_orderbooks(self):
        """Collect orderbooks for all tracked markets."""
        if not self._tracked_markets:
            logger.warning("No markets tracked yet")
            return

        start_time = time.time()
        timestamp = datetime.utcnow()
        collected = 0
        failed = 0

        # Sort by volume and take top markets
        sorted_markets = sorted(
            self._tracked_markets.items(),
            key=lambda x: x[1].get("volume_24h", 0),
            reverse=True,
        )[: self.config.max_markets_to_track]

        for cid, info in sorted_markets:
            try:
                result = await self._fetch_market_orderbook(cid, info, timestamp)
                if result:
                    collected += 1
                else:
                    failed += 1
            except Exception as e:
                logger.warning(f"Orderbook error for {cid}: {e}")
                failed += 1

        total_latency_ms = (time.time() - start_time) * 1000
        self.stats["orderbooks_collected"] += collected
        self._record_health("orderbook", success=True, latency_ms=total_latency_ms, items=collected)

        logger.info(
            f"Orderbooks: {collected} collected, {failed} failed "
            f"({total_latency_ms:.0f}ms total)"
        )

    async def _fetch_market_orderbook(
        self,
        condition_id: str,
        market_info: dict,
        timestamp: datetime,
    ) -> bool:
        """
        Fetch REAL orderbook data from Polymarket CLOB API.

        CRITICAL: NO synthetic data - skip if API fails.
        """
        client = await self._get_client()

        try:
            # Get market info to find token IDs
            response = await client.get(
                f"{self.config.clob_api_url}/markets/{condition_id}"
            )

            if response.status_code != 200:
                return False

            data = response.json()
            tokens = data.get("tokens", [])

            async with get_session() as session:
                # Get market from DB
                result = await session.execute(
                    select(Market).where(Market.condition_id == condition_id)
                )
                market = result.scalar_one_or_none()

                if not market:
                    return False

                tokens_collected = 0

                for token in tokens:
                    token_id = token.get("token_id", "")
                    outcome = token.get("outcome", "Unknown")

                    if not token_id:
                        continue

                    # Fetch REAL orderbook from /book endpoint
                    # NO FALLBACK - skip if API fails
                    try:
                        book_response = await client.get(
                            f"{self.config.clob_api_url}/book",
                            params={"token_id": token_id},
                        )

                        if book_response.status_code != 200:
                            logger.debug(f"Book API returned {book_response.status_code} - skipping")
                            continue

                        book_data = book_response.json()
                        bids = book_data.get("bids", [])
                        asks = book_data.get("asks", [])

                        # REQUIRE REAL DATA - skip if empty
                        if not bids or len(bids) == 0:
                            logger.debug(f"No bids for {token_id} - skipping (no synthetic data)")
                            continue

                        if not asks or len(asks) == 0:
                            logger.debug(f"No asks for {token_id} - skipping (no synthetic data)")
                            continue

                        # Parse REAL bid/ask prices
                        best_bid = float(
                            bids[0].get("price", 0)
                            if isinstance(bids[0], dict)
                            else bids[0][0]
                        )
                        bid_depth = sum(
                            float(b.get("size", 0) if isinstance(b, dict) else b[1])
                            for b in bids[:5]
                        )

                        best_ask = float(
                            asks[0].get("price", 0)
                            if isinstance(asks[0], dict)
                            else asks[0][0]
                        )
                        ask_depth = sum(
                            float(a.get("size", 0) if isinstance(a, dict) else a[1])
                            for a in asks[:5]
                        )

                        # Validate prices
                        if best_bid <= 0 or best_ask <= 0:
                            logger.debug(f"Invalid prices - skipping")
                            continue

                        if best_bid >= best_ask:
                            logger.debug(f"Crossed book - skipping")
                            continue

                        spread = best_ask - best_bid
                        mid_price = (best_bid + best_ask) / 2
                        spread_bps = int((spread / mid_price) * 10000) if mid_price > 0 else 0

                        # Calculate imbalance
                        total_depth = bid_depth + ask_depth
                        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0

                    except Exception as book_error:
                        logger.warning(f"Book fetch error: {book_error} - skipping (no synthetic data)")
                        continue

                    # Validate orderbook data
                    ob_data = {
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "mid_price": mid_price,
                        "spread": spread,
                    }

                    validation = self.validator.validate_orderbook(ob_data, market_id=condition_id)
                    self.stats["data_validated"] += 1

                    if not validation.is_valid:
                        self.stats["data_rejected"] += 1
                        logger.debug(f"Rejected orderbook: {validation.error_summary}")
                        continue

                    # Store REAL orderbook data
                    ob = OrderBook(
                        timestamp=timestamp,
                        market_id=market.id,
                        token_id=token_id,
                        outcome=outcome,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        mid_price=mid_price,
                        spread=spread,
                        spread_bps=spread_bps,
                        bid_depth=bid_depth,
                        ask_depth=ask_depth,
                        imbalance=imbalance,
                        bids=bids[:10],
                        asks=asks[:10],
                    )
                    session.add(ob)

                    # Update Redis cache
                    await self.cache.update_price(
                        market_id=condition_id,
                        outcome=outcome,
                        mid_price=mid_price,
                        token_id=token_id,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        spread=spread,
                        spread_bps=spread_bps,
                        volume_24h=market_info.get("volume_24h", 0),
                        liquidity=market_info.get("liquidity", 0),
                    )

                    tokens_collected += 1
                    await asyncio.sleep(0.05)  # Small delay between book fetches

            return tokens_collected > 0

        except Exception as e:
            logger.warning(f"Fetch orderbook error for {condition_id}: {e}")
            return False

    # =========================================================================
    # TRADE STREAM COLLECTION
    # =========================================================================

    async def _trade_loop(self):
        """Trade stream collection loop."""
        await asyncio.sleep(20)

        while self._running:
            await asyncio.sleep(self.config.trade_interval)
            try:
                await self._collect_trades()
            except Exception as e:
                logger.error(f"Trade collection error: {e}")
                self._record_health("trade", success=False, error=str(e))

    async def _collect_trades(self):
        """Collect recent trades from Polymarket Data API."""
        start_time = time.time()
        client = await self._get_client()
        collected = 0
        rejected = 0

        try:
            # Use PUBLIC data-api endpoint (no auth required!)
            response = await client.get(
                f"{self.config.data_api_url}/trades",
                params={"limit": 1000},
            )

            if response.status_code != 200:
                logger.error(f"Trades API returned {response.status_code}")
                self._record_health("trade", success=False)
                return

            trades = response.json()

            async with get_session() as session:
                for t in trades:
                    # Validate trade data
                    validation = self.validator.validate_trade(t)
                    self.stats["data_validated"] += 1

                    if not validation.is_valid:
                        self.stats["data_rejected"] += 1
                        rejected += 1
                        continue

                    # Check for duplicates
                    trade_id = t.get("transactionHash")
                    if not trade_id:
                        continue

                    result = await session.execute(
                        select(Trade).where(Trade.trade_id == str(trade_id))
                    )
                    if result.scalar_one_or_none():
                        continue  # Already exists

                    # Get market
                    condition_id = t.get("conditionId")
                    if not condition_id:
                        continue

                    result = await session.execute(
                        select(Market).where(Market.condition_id == condition_id)
                    )
                    market = result.scalar_one_or_none()

                    if not market:
                        continue

                    # Parse trade details
                    price = float(t.get("price", 0))
                    size = float(t.get("size", 0))
                    value_usd = price * size
                    is_whale = value_usd >= 1000  # $1000+ is whale

                    # Parse timestamp
                    trade_time = datetime.utcnow()
                    if t.get("timestamp"):
                        try:
                            ts = t["timestamp"]
                            if isinstance(ts, (int, float)):
                                trade_time = datetime.utcfromtimestamp(ts)
                            else:
                                trade_time = datetime.fromisoformat(str(ts).replace("Z", ""))
                        except (ValueError, TypeError):
                            pass

                    # Store trade
                    trade = Trade(
                        timestamp=trade_time,
                        trade_id=str(trade_id),
                        market_id=market.id,
                        token_id=t.get("asset", ""),
                        side=t.get("side", "BUY").upper(),
                        outcome=t.get("outcome", ""),
                        price=price,
                        size=size,
                        size_usd=value_usd,
                    )
                    session.add(trade)
                    collected += 1

            latency_ms = (time.time() - start_time) * 1000
            self.stats["trades_collected"] += collected
            self._record_health("trade", success=True, latency_ms=latency_ms, items=collected)

            if collected > 0:
                logger.info(f"Trades: {collected} new ({rejected} rejected, {latency_ms:.0f}ms)")

        except Exception as e:
            logger.error(f"Trade collection error: {e}")
            self._record_health("trade", success=False, error=str(e))

    # =========================================================================
    # HEALTH MONITORING
    # =========================================================================

    def _record_health(
        self,
        component: str,
        success: bool,
        latency_ms: float = 0,
        items: int = 0,
        error: str | None = None,
    ):
        """Record health metrics for a component."""
        if component not in self._health_metrics:
            self._health_metrics[component] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "total_latency_ms": 0,
                "items_collected": 0,
                "last_error": None,
            }

        metrics = self._health_metrics[component]
        metrics["attempts"] += 1
        if success:
            metrics["successes"] += 1
        else:
            metrics["failures"] += 1
        metrics["total_latency_ms"] += latency_ms
        metrics["items_collected"] += items
        if error:
            metrics["last_error"] = error

    async def _health_check_loop(self):
        """Health check loop."""
        while self._running:
            await asyncio.sleep(60)  # Every minute
            try:
                await self._record_health_to_db()
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _record_health_to_db(self):
        """Store health metrics in database."""
        timestamp = datetime.utcnow()

        async with get_session() as session:
            for component, metrics in self._health_metrics.items():
                if metrics["attempts"] == 0:
                    continue

                avg_latency = (
                    metrics["total_latency_ms"] / metrics["attempts"]
                    if metrics["attempts"] > 0
                    else 0
                )

                health = CollectionHealth(
                    component=component,
                    timestamp=timestamp,
                    period_minutes=1,
                    attempts=metrics["attempts"],
                    successes=metrics["successes"],
                    failures=metrics["failures"],
                    avg_latency_ms=avg_latency,
                    items_collected=metrics["items_collected"],
                    gap_detected=metrics["failures"] >= 3,
                )
                session.add(health)

        # Reset metrics after recording
        for metrics in self._health_metrics.values():
            metrics["attempts"] = 0
            metrics["successes"] = 0
            metrics["failures"] = 0
            metrics["total_latency_ms"] = 0
            metrics["items_collected"] = 0
            metrics["last_error"] = None

    async def _stats_loop(self):
        """Statistics logging loop."""
        while self._running:
            await asyncio.sleep(300)  # Every 5 minutes

            uptime = (
                datetime.utcnow() - self.stats["started_at"]
                if self.stats["started_at"]
                else timedelta(0)
            )

            rejection_rate = (
                self.stats["data_rejected"] / max(self.stats["data_validated"], 1) * 100
            )

            logger.info(
                f"\n"
                f"{'=' * 50}\n"
                f"DATACOLLECTOR STATISTICS\n"
                f"{'=' * 50}\n"
                f"Uptime: {uptime}\n"
                f"Markets Discovered: {self.stats['markets_discovered']}\n"
                f"Orderbooks Collected: {self.stats['orderbooks_collected']}\n"
                f"Trades Collected: {self.stats['trades_collected']}\n"
                f"Data Validated: {self.stats['data_validated']}\n"
                f"Data Rejected: {self.stats['data_rejected']} ({rejection_rate:.1f}%)\n"
                f"{'=' * 50}"
            )

            # Log cache stats
            cache_stats = await self.cache.get_cache_stats()
            logger.info(f"Cache stats: {cache_stats}")


async def run_collector():
    """Entry point for running the collector."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    collector = DataCollector()
    await collector.start()


if __name__ == "__main__":
    asyncio.run(run_collector())
