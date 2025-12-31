"""
Arbitrage Strategy.

Detects price sum deviations where buying all outcomes costs less than $1.

Example:
    YES = $0.48, NO = $0.49
    Total = $0.97 (3% gap)
    Buy both -> guaranteed $1.00 payout -> 3% profit
"""

import logging
import time
from datetime import datetime
from typing import Any

from polybot.cache import RedisCache, MarketPrices
from polybot.models import SignalAction
from polybot.strategies.base import BaseStrategy, Signal, StrategyResult

logger = logging.getLogger(__name__)


class ArbitrageStrategy(BaseStrategy):
    """
    Price Sum Arbitrage Strategy.

    Looks for markets where:
    - Sum of all outcome prices < 1.00 (underpriced - guaranteed profit)
    - Sum of all outcome prices > 1.00 (overpriced - theoretical sell)
    """

    def __init__(
        self,
        min_gap: float = 0.005,  # 0.5% minimum gap
        max_gap: float = 0.15,   # 15% maximum (filter suspicious data)
        min_liquidity: float = 1000,  # $1000 minimum liquidity
    ):
        super().__init__()
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.min_liquidity = min_liquidity

    @property
    def name(self) -> str:
        return "arbitrage"

    def get_scan_interval(self) -> int:
        return 30  # Every 30 seconds (arbitrage is time-sensitive)

    async def scan(self, cache: RedisCache) -> StrategyResult:
        """
        Scan for arbitrage opportunities.

        Uses Redis cache for fast scanning (<1ms vs 200-500ms per HTTP request).
        """
        start_time = time.time()
        signals = []
        errors = []
        markets_scanned = 0
        opportunities_found = 0
        near_misses = 0

        try:
            # Get all markets from cache
            markets = await cache.get_all_markets()
            markets_scanned = len(markets)

            logger.info(f"Scanning {markets_scanned} markets for arbitrage...")

            for market in markets:
                try:
                    gap = market.arbitrage_gap
                    slug = market.slug[:30] if market.slug else "unknown"

                    # Track near-misses (1-2% gap)
                    if 0.01 <= gap < self.min_gap:
                        near_misses += 1
                        logger.debug(f"Near-miss: {slug} gap={gap:.2%}")

                    # Check if gap is significant
                    if gap < self.min_gap:
                        continue

                    # Check if gap is suspiciously large
                    if gap > self.max_gap:
                        logger.warning(f"Large gap in {slug}: {gap:.2%} - possible data error")
                        continue

                    # Check liquidity
                    if market.liquidity < self.min_liquidity:
                        logger.debug(f"Low liquidity: {slug} ${market.liquidity:,.0f}")
                        continue

                    # Create signal for underpriced market
                    if market.is_underpriced:
                        signal = self._create_arbitrage_signal(market, gap)
                        if signal and self.validate_signal(signal):
                            signals.append(signal)
                            opportunities_found += 1
                            logger.info(f"Arbitrage found: {slug} gap={gap:.2%}")

                except Exception as e:
                    errors.append(f"Error analyzing {market.slug}: {e}")

        except Exception as e:
            errors.append(f"Scan error: {e}")
            logger.error(f"Arbitrage scan error: {e}")

        # Update stats
        self.last_scan = datetime.utcnow()
        self.total_scans += 1
        self.total_signals_generated += len(signals)

        scan_duration_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Arbitrage scan complete: {markets_scanned} markets, "
            f"{opportunities_found} opportunities, {near_misses} near-misses "
            f"({scan_duration_ms:.0f}ms)"
        )

        return StrategyResult(
            signals=signals,
            markets_scanned=markets_scanned,
            scan_duration_ms=scan_duration_ms,
            errors=errors,
            metadata={
                "opportunities_found": opportunities_found,
                "near_misses": near_misses,
                "min_gap_threshold": self.min_gap,
            },
        )

    def _create_arbitrage_signal(
        self,
        market: MarketPrices,
        gap: float,
    ) -> Signal | None:
        """Create a BUY_ALL signal for underpriced market."""
        try:
            # Calculate confidence based on gap size and liquidity
            gap_confidence = min(1.0, gap / 0.05)  # 5% gap = full confidence
            liquidity_factor = min(1.0, market.liquidity / 100000)  # $100K = full confidence
            confidence = gap_confidence * (0.7 + 0.3 * liquidity_factor)

            price_sum = market.sum_price

            return self.create_signal(
                market_id=market.condition_id,
                market_slug=market.slug,
                token_id=market.yes_token_id,  # Primary token
                action=SignalAction.BUY_ALL,
                confidence=confidence,
                expected_edge=gap,
                entry_price=price_sum,
                exit_price=1.0,  # Guaranteed payout
                reasoning=(
                    f"Price sum arbitrage: Yes=${market.yes_price:.3f} + "
                    f"No=${market.no_price:.3f} = ${price_sum:.4f} (gap: {gap:.2%})"
                ),
                metrics={
                    "yes_price": market.yes_price,
                    "no_price": market.no_price,
                    "price_sum": price_sum,
                    "gap": gap,
                    "gap_pct": gap * 100,
                    "liquidity": market.liquidity,
                    "spread": market.spread,
                },
                expires_in_minutes=5,  # Arbitrage expires quickly
            )

        except Exception as e:
            logger.error(f"Error creating arbitrage signal: {e}")
            return None

    def check_arbitrage(
        self,
        market_id: int,
        prices: dict[str, float],
    ) -> dict | None:
        """
        Check for arbitrage opportunity (for backtesting).

        Args:
            market_id: Market ID
            prices: Dict of outcome -> price

        Returns:
            Signal dict if opportunity found, None otherwise
        """
        if not prices:
            return None

        yes_price = prices.get("Yes", 0.5)
        no_price = prices.get("No", 0.5)
        price_sum = yes_price + no_price
        gap = abs(1.0 - price_sum)

        if gap < self.min_gap:
            return None

        if gap > self.max_gap:
            return None

        if price_sum < 1.0:  # Underpriced
            return {
                "market_id": market_id,
                "outcome": "Yes",  # Buy both, but track as Yes
                "action": "BUY_ALL",
                "confidence": min(1.0, gap / 0.05) * 0.8,
                "entry_price": price_sum,
                "gap": gap,
            }

        return None

    def analyze_market(self, prices: dict[str, float]) -> dict[str, Any]:
        """
        Analyze a single market for arbitrage.

        Args:
            prices: Dict of outcome -> price

        Returns:
            Analysis results
        """
        if not prices:
            return {"has_opportunity": False, "reason": "No prices"}

        price_sum = sum(prices.values())
        gap = abs(1.0 - price_sum)

        return {
            "has_opportunity": gap >= self.min_gap,
            "price_sum": price_sum,
            "gap": gap,
            "gap_pct": gap * 100,
            "type": "underpriced" if price_sum < 1.0 else "overpriced",
            "prices": prices,
        }
