"""
Database connection and session management.

Uses async SQLAlchemy with asyncpg driver for PostgreSQL + TimescaleDB.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from polybot.config import get_config

logger = logging.getLogger(__name__)

# Global engine instance
_engine = None
_session_factory = None


def get_engine():
    """Get or create the async database engine."""
    global _engine
    if _engine is None:
        config = get_config()
        _engine = create_async_engine(
            config.database_url,
            echo=False,  # Set True for SQL logging
            poolclass=NullPool,  # Better for async
            connect_args={
                "server_settings": {
                    "application_name": "polybot",
                }
            },
        )
        logger.info("Database engine created")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get the session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session with automatic commit/rollback.

    Usage:
        async with get_session() as session:
            result = await session.execute(select(Market))
            markets = result.scalars().all()
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_db():
    """Initialize database connection and verify connectivity."""
    engine = get_engine()
    try:
        async with engine.begin() as conn:
            # Simple connectivity test
            await conn.execute("SELECT 1")
        logger.info("Database connection verified")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise


async def close_db():
    """Close database connections."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connections closed")
