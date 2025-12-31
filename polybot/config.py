"""
Centralized configuration from environment variables.

All settings are loaded from environment variables with sensible defaults.
Use .env file for local development.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # Database
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://polybot:polybot_dev_password@localhost:5432/polybot",
        )
    )

    # Redis
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )

    # Polymarket APIs
    gamma_api_url: str = field(
        default_factory=lambda: os.getenv(
            "GAMMA_API_URL", "https://gamma-api.polymarket.com"
        )
    )
    clob_api_url: str = field(
        default_factory=lambda: os.getenv("CLOB_API_URL", "https://clob.polymarket.com")
    )
    data_api_url: str = field(
        default_factory=lambda: os.getenv(
            "DATA_API_URL", "https://data-api.polymarket.com"
        )
    )

    # Collection intervals (seconds)
    orderbook_interval: int = field(
        default_factory=lambda: int(os.getenv("ORDERBOOK_INTERVAL", "30"))
    )
    market_refresh_interval: int = field(
        default_factory=lambda: int(os.getenv("MARKET_REFRESH_INTERVAL", "300"))
    )
    trade_interval: int = field(
        default_factory=lambda: int(os.getenv("TRADE_INTERVAL", "30"))
    )

    # Collection limits
    max_markets_to_track: int = field(
        default_factory=lambda: int(os.getenv("MAX_MARKETS_TO_TRACK", "100"))
    )
    request_delay: float = field(
        default_factory=lambda: float(os.getenv("REQUEST_DELAY", "0.15"))
    )

    # LLM (for AI Analyst strategy)
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openrouter")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "anthropic/claude-3.5-sonnet")
    )
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))

    # API
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(
        default_factory=lambda: int(os.getenv("API_PORT", "8000"))
    )

    # Rate limiting
    rate_limit_free: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_FREE", "100"))
    )
    rate_limit_paid: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_PAID", "1000"))
    )

    # Stripe (Phase 3)
    stripe_secret_key: str = field(
        default_factory=lambda: os.getenv("STRIPE_SECRET_KEY", "")
    )
    stripe_webhook_secret: str = field(
        default_factory=lambda: os.getenv("STRIPE_WEBHOOK_SECRET", "")
    )

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "json"))

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.database_url:
            errors.append("DATABASE_URL is required")

        if not self.redis_url:
            errors.append("REDIS_URL is required")

        if self.orderbook_interval < 10:
            errors.append("ORDERBOOK_INTERVAL must be at least 10 seconds")

        if self.max_markets_to_track < 1:
            errors.append("MAX_MARKETS_TO_TRACK must be at least 1")

        return errors


@lru_cache()
def get_config() -> Config:
    """Get cached configuration instance."""
    config = Config()
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    return config
