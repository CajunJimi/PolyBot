#!/usr/bin/env python
"""
Run the PolyBot data collector.

Usage:
    python scripts/run_collector.py
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from polybot.collector import DataCollector


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


async def main():
    """Main entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting PolyBot Data Collector...")
    logger.info("Press Ctrl+C to stop")

    collector = DataCollector()

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def shutdown():
        logger.info("Shutdown signal received")
        asyncio.create_task(collector.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)

    try:
        await collector.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await collector.stop()
        logger.info("Collector stopped")


if __name__ == "__main__":
    asyncio.run(main())
