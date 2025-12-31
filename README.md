# PolyBot

Polymarket Trading & Data Platform - Built for reliability and simplicity.

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Docker & Docker Compose

### 2. Setup

```bash
# Clone and enter directory
cd "PolyBot App"

# Copy environment template
cp .env.example .env

# Start PostgreSQL + Redis
docker-compose up -d

# Install Python dependencies
pip install -e ".[dev]"
```

### 3. Run Data Collector

```bash
python scripts/run_collector.py
```

### 4. Run Dashboard

```bash
streamlit run dashboard/app.py
```

## Project Structure

```
polybot/
├── config.py       # Configuration from env vars
├── db.py           # Database connection
├── models.py       # ORM models
├── collector.py    # Data collector (markets, orderbooks, trades)
├── validator.py    # Data validation before storage
├── cache.py        # Redis cache for strategies
├── backtest/       # Backtest framework
└── strategies/     # Trading strategies
```

## Key Principles

1. **No fake data** - Skip if API fails, never generate synthetic prices
2. **Validate first** - All data validated before storage
3. **Single source of truth** - One collector, one database
4. **Test before deploy** - Backtest → Paper trade → Live

## API Endpoints (Polymarket)

- Markets: `https://gamma-api.polymarket.com/markets`
- Orderbooks: `https://clob.polymarket.com/book?token_id={id}`
- Trades: `https://data-api.polymarket.com/trades` (PUBLIC)

## Verify Data Collection

```bash
# Check database has data
docker exec -it polybot-postgres psql -U polybot -c "SELECT COUNT(*) FROM orderbooks;"
```

## License

MIT
