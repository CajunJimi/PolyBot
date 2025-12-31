"""
AI Analyst Strategy.

Uses LLM (Claude/GPT) to analyze markets and generate trading signals.

Features:
- Multi-provider support (Anthropic, OpenAI, OpenRouter)
- Priority scoring for market selection
- Price history and momentum context
- JSON response parsing
"""

import json
import logging
import time
from datetime import datetime
from typing import Any

import httpx

from polybot.cache import RedisCache, MarketPrices
from polybot.config import get_config
from polybot.models import SignalAction
from polybot.strategies.base import BaseStrategy, Signal, StrategyResult

logger = logging.getLogger(__name__)


class AIAnalystStrategy(BaseStrategy):
    """
    AI-powered market analysis strategy.

    Uses LLM to:
    - Identify upcoming catalysts
    - Assess sentiment vs reality
    - Find mispriced markets
    """

    def __init__(
        self,
        min_confidence: float = 0.65,
        max_markets_per_scan: int = 12,
    ):
        super().__init__()
        self.min_confidence = min_confidence
        self.max_markets_per_scan = max_markets_per_scan

        # Get LLM config
        config = get_config()
        self.llm_provider = config.llm_provider.lower()
        self.llm_model = config.llm_model
        self.llm_api_key = config.llm_api_key

        # Setup provider
        self._setup_provider()

        logger.info(f"AI Analyst initialized: provider={self.llm_provider}, model={self.llm_model}")

    def _setup_provider(self):
        """Configure LLM provider and model."""
        if self.llm_provider == "openrouter":
            if not self.llm_model or "haiku" in self.llm_model.lower():
                self.llm_model = "anthropic/claude-3.5-sonnet"
        elif self.llm_provider == "anthropic":
            model = self.llm_model.split("/")[-1] if "/" in self.llm_model else self.llm_model
            if not model or "haiku" in model.lower():
                model = "claude-3-5-sonnet-20241022"
            self.llm_model = model
        elif self.llm_provider == "openai":
            model = self.llm_model.split("/")[-1] if "/" in self.llm_model else self.llm_model
            if not model:
                model = "gpt-4o"
            self.llm_model = model
        else:
            # Default to OpenRouter
            self.llm_provider = "openrouter"
            self.llm_model = "anthropic/claude-3.5-sonnet"

    @property
    def name(self) -> str:
        return "ai_analyst"

    def get_scan_interval(self) -> int:
        return 300  # Every 5 minutes

    async def scan(self, cache: RedisCache) -> StrategyResult:
        """Scan markets using AI analysis."""
        start_time = time.time()
        signals = []
        errors = []
        markets_scanned = 0
        ai_recommendations = 0

        if not self.llm_api_key:
            logger.warning("AI Analyst: No LLM_API_KEY set, skipping")
            return StrategyResult(
                signals=[],
                markets_scanned=0,
                scan_duration_ms=0,
                errors=["No API key configured"],
            )

        try:
            # Get markets with filtering
            markets = await self._get_priority_markets(cache)
            markets_scanned = len(markets)

            logger.info(f"AI Analyst scanning {markets_scanned} priority markets...")

            if markets_scanned == 0:
                return StrategyResult(
                    signals=[],
                    markets_scanned=0,
                    scan_duration_ms=(time.time() - start_time) * 1000,
                    errors=[],
                )

            # Prepare market data for AI
            market_summaries = self._prepare_market_data(markets)

            # Get AI analysis
            ai_response = await self._get_ai_analysis(market_summaries)

            if ai_response:
                recommendations = self._parse_ai_response(ai_response, markets)

                for rec in recommendations:
                    if rec["confidence"] >= self.min_confidence:
                        signal = self._create_ai_signal(rec)
                        if signal and self.validate_signal(signal):
                            signals.append(signal)
                            ai_recommendations += 1
                            logger.info(
                                f"AI Signal: {rec['market_slug'][:30]} "
                                f"{rec['action']} @ {rec['confidence']:.0%}"
                            )

        except Exception as e:
            errors.append(f"AI scan error: {e}")
            logger.error(f"AI Analyst error: {e}")

        scan_duration_ms = (time.time() - start_time) * 1000

        logger.info(
            f"AI Analyst complete: {markets_scanned} markets, "
            f"{ai_recommendations} recommendations ({scan_duration_ms:.0f}ms)"
        )

        # Update stats
        self.last_scan = datetime.utcnow()
        self.total_scans += 1
        self.total_signals_generated += len(signals)

        return StrategyResult(
            signals=signals,
            markets_scanned=markets_scanned,
            scan_duration_ms=scan_duration_ms,
            errors=errors,
            metadata={"ai_recommendations": ai_recommendations},
        )

    async def _get_priority_markets(self, cache: RedisCache) -> list[MarketPrices]:
        """Get markets prioritized by trading opportunity potential."""
        all_markets = await cache.get_all_markets()

        scored_markets = []
        for m in all_markets:
            # Basic filters
            if m.volume_24h < 5000 or m.liquidity < 3000:
                continue

            # Skip extreme prices
            if m.yes_price < 0.10 or m.yes_price > 0.90:
                continue

            # Calculate priority score
            score = 0

            # Volume score (0-30)
            score += min(30, m.volume_24h / 50000 * 30)

            # Liquidity score (0-20)
            score += min(20, m.liquidity / 20000 * 20)

            # Price range score - prefer 20-80% range (0-20)
            if 0.20 <= m.yes_price <= 0.80:
                score += 20
            elif 0.15 <= m.yes_price <= 0.85:
                score += 10

            scored_markets.append((m, score))

        # Sort by score
        scored_markets.sort(key=lambda x: x[1], reverse=True)

        return [m for m, _ in scored_markets[: self.max_markets_per_scan]]

    def _prepare_market_data(self, markets: list[MarketPrices]) -> str:
        """Format market data for AI consumption."""
        summaries = []

        for i, m in enumerate(markets, 1):
            summary = f"""
Market {i}: {m.slug}
- YES: {m.yes_price:.1%} | NO: {m.no_price:.1%}
- Volume 24h: ${m.volume_24h:,.0f}
- Liquidity: ${m.liquidity:,.0f}
- Spread: {m.spread:.4f}"""
            summaries.append(summary)

        return "\n".join(summaries)

    async def _get_ai_analysis(self, market_data: str) -> str | None:
        """Call AI API for market analysis."""
        current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        system_prompt = f"""You are an expert prediction market trader on Polymarket.
Current time: {current_time}

YOUR MISSION: Identify HIGH-CONVICTION trading opportunities.

ANALYSIS FRAMEWORK:
1. CATALYST IDENTIFICATION - What events could move this market?
2. SENTIMENT vs REALITY - Is current price justified?
3. RISK/REWARD - Minimum 1.5:1 ratio required

TRADING RULES:
- Only recommend trades with 65%+ confidence
- Prefer markets with clear catalysts
- Consider liquidity for exit

Respond in JSON format:
{{
  "analysis_summary": "Brief overview",
  "recommendations": [
    {{
      "market_slug": "exact-slug-from-input",
      "action": "BUY_YES" or "BUY_NO" or "SKIP",
      "confidence": 0.65 to 1.0,
      "reasoning": "Why this trade makes sense",
      "risk": "What could go wrong"
    }}
  ]
}}

Be selective - 2-4 strong picks is better than 10 weak ones."""

        user_prompt = f"""TRADING SCAN - {current_time}

MARKETS TO ANALYZE:
{market_data}

Find the BEST 2-5 trading opportunities. Skip markets without clear edge."""

        try:
            if self.llm_provider == "openrouter":
                return await self._call_openrouter(system_prompt, user_prompt)
            elif self.llm_provider == "anthropic":
                return await self._call_anthropic(system_prompt, user_prompt)
            else:
                return await self._call_openai(system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"AI API error: {e}")
            return None

    async def _call_anthropic(self, system: str, user: str) -> str | None:
        """Call Anthropic Claude API."""
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.llm_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.llm_model,
                    "max_tokens": 2000,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )

            if response.status_code == 200:
                data = response.json()
                return data["content"][0]["text"]
            else:
                logger.error(f"Anthropic API error: {response.status_code}")
                return None

    async def _call_openai(self, system: str, user: str) -> str | None:
        """Call OpenAI GPT API."""
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.7,
                },
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                logger.error(f"OpenAI API error: {response.status_code}")
                return None

    async def _call_openrouter(self, system: str, user: str) -> str | None:
        """Call OpenRouter API."""
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://polybot.app",
                    "X-Title": "PolyBot",
                },
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.7,
                },
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                logger.error(f"OpenRouter API error: {response.status_code}")
                return None

    def _parse_ai_response(
        self,
        response: str,
        markets: list[MarketPrices],
    ) -> list[dict]:
        """Parse AI response into recommendations."""
        recommendations = []

        try:
            # Find JSON in response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                # Create market lookup
                market_lookup = {m.slug: m for m in markets}

                for rec in data.get("recommendations", []):
                    slug = rec.get("market_slug", "")
                    action = rec.get("action", "SKIP").upper()

                    if action in ["SKIP", "HOLD"]:
                        continue

                    # Find matching market
                    market = market_lookup.get(slug)
                    if not market:
                        # Try fuzzy match
                        for m_slug in market_lookup:
                            if slug in m_slug or m_slug in slug:
                                market = market_lookup[m_slug]
                                break

                    if not market:
                        logger.warning(f"Market not found: {slug}")
                        continue

                    recommendations.append({
                        "market": market,
                        "market_slug": market.slug,
                        "action": action,
                        "confidence": float(rec.get("confidence", 0)),
                        "reasoning": rec.get("reasoning", ""),
                        "risk": rec.get("risk", ""),
                    })

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI JSON: {e}")
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")

        return recommendations

    def _create_ai_signal(self, rec: dict) -> Signal | None:
        """Create signal from AI recommendation."""
        try:
            market: MarketPrices = rec["market"]

            if rec["action"] == "BUY_YES":
                action = SignalAction.BUY_YES
                entry_price = market.yes_price
                token_id = market.yes_token_id
            else:
                action = SignalAction.BUY_NO
                entry_price = market.no_price
                token_id = market.no_token_id

            return self.create_signal(
                market_id=market.condition_id,
                market_slug=market.slug,
                token_id=token_id,
                action=action,
                confidence=rec["confidence"],
                expected_edge=max(0.05, rec["confidence"] - 0.5),
                entry_price=entry_price,
                reasoning=rec["reasoning"],
                metrics={
                    "ai_confidence": rec["confidence"],
                    "risk": rec["risk"],
                    "model": self.llm_model,
                },
                expires_in_minutes=60,  # AI signals last longer
            )

        except Exception as e:
            logger.error(f"Error creating AI signal: {e}")
            return None
