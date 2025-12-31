-- PolyBot Database Initialization
-- This script runs automatically when the PostgreSQL container starts

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Markets table
CREATE TABLE IF NOT EXISTS markets (
    id SERIAL PRIMARY KEY,
    condition_id VARCHAR(100) UNIQUE NOT NULL,
    slug VARCHAR(200) NOT NULL,
    question TEXT,
    description TEXT,
    category VARCHAR(100),
    tokens JSONB,
    active BOOLEAN DEFAULT TRUE,
    closed BOOLEAN DEFAULT FALSE,
    resolved BOOLEAN DEFAULT FALSE,
    resolution VARCHAR(50),
    end_date TIMESTAMPTZ,
    volume_24h DECIMAL(18,2) DEFAULT 0,
    liquidity DECIMAL(18,2) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_markets_active_volume ON markets (active, volume_24h DESC);
CREATE INDEX IF NOT EXISTS idx_markets_condition_id ON markets (condition_id);
CREATE INDEX IF NOT EXISTS idx_markets_slug ON markets (slug);

-- Orderbooks table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS orderbooks (
    time TIMESTAMPTZ NOT NULL,
    market_id INTEGER NOT NULL REFERENCES markets(id) ON DELETE CASCADE,
    token_id VARCHAR(100) NOT NULL,
    outcome VARCHAR(50),
    best_bid DECIMAL(8,6),
    best_ask DECIMAL(8,6),
    mid_price DECIMAL(8,6),
    spread DECIMAL(8,6),
    spread_bps INTEGER,
    bid_depth DECIMAL(18,2),
    ask_depth DECIMAL(18,2),
    imbalance DECIMAL(4,3),
    bids JSONB,
    asks JSONB
);

-- Convert to hypertable (time-series optimized)
SELECT create_hypertable('orderbooks', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_orderbooks_market_time ON orderbooks (market_id, time DESC);

-- Trades table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS trades (
    time TIMESTAMPTZ NOT NULL,
    trade_id VARCHAR(100) UNIQUE NOT NULL,
    market_id INTEGER NOT NULL REFERENCES markets(id) ON DELETE CASCADE,
    condition_id VARCHAR(100),
    token_id VARCHAR(100),
    side VARCHAR(4) NOT NULL,
    outcome VARCHAR(50),
    price DECIMAL(8,6) NOT NULL,
    size DECIMAL(18,6) NOT NULL,
    value_usd DECIMAL(18,2),
    maker_address VARCHAR(100),
    taker_address VARCHAR(100),
    is_whale BOOLEAN DEFAULT FALSE
);

SELECT create_hypertable('trades', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_trades_market_time ON trades (market_id, time DESC);

-- Backtest results table
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    strategy_config JSONB,
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ NOT NULL,
    initial_balance DECIMAL(18,2) DEFAULT 10000,
    final_balance DECIMAL(18,2),
    total_return DECIMAL(18,2),
    total_return_pct DECIMAL(8,4),
    sharpe_ratio DECIMAL(6,3),
    sortino_ratio DECIMAL(6,3),
    max_drawdown_pct DECIMAL(6,3),
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5,2),
    profit_factor DECIMAL(6,3),
    avg_win DECIMAL(18,2),
    avg_loss DECIMAL(18,2),
    trade_details JSONB,
    equity_curve JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results (strategy_name, created_at DESC);

-- Signals table
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    market_id INTEGER REFERENCES markets(id),
    market_slug VARCHAR(200),
    token_id VARCHAR(100),
    action VARCHAR(20) NOT NULL,
    confidence DECIMAL(4,3) NOT NULL,
    expected_edge DECIMAL(6,4),
    entry_price DECIMAL(8,6),
    exit_price DECIMAL(8,6),
    reasoning TEXT,
    metrics JSONB,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    executed_at TIMESTAMPTZ,
    result JSONB
);

CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals (strategy_name, created_at DESC);

-- Collection health table
CREATE TABLE IF NOT EXISTS collection_health (
    id SERIAL PRIMARY KEY,
    component VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    period_minutes INTEGER DEFAULT 1,
    attempts INTEGER DEFAULT 0,
    successes INTEGER DEFAULT 0,
    failures INTEGER DEFAULT 0,
    avg_latency_ms DECIMAL(10,2),
    items_collected INTEGER DEFAULT 0,
    gap_detected BOOLEAN DEFAULT FALSE,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_health_component_time ON collection_health (component, timestamp DESC);

-- TimescaleDB compression policies (compress data older than 7 days)
SELECT add_compression_policy('orderbooks', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('trades', INTERVAL '7 days', if_not_exists => TRUE);

-- TimescaleDB retention policies (keep 90 days of data)
SELECT add_retention_policy('orderbooks', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('trades', INTERVAL '90 days', if_not_exists => TRUE);

-- Useful views
CREATE OR REPLACE VIEW v_market_prices AS
SELECT
    m.id,
    m.condition_id,
    m.slug,
    m.question,
    m.volume_24h,
    m.liquidity,
    o.time as price_time,
    o.outcome,
    o.mid_price,
    o.best_bid,
    o.best_ask,
    o.spread
FROM markets m
LEFT JOIN LATERAL (
    SELECT * FROM orderbooks
    WHERE market_id = m.id
    ORDER BY time DESC
    LIMIT 2
) o ON true
WHERE m.active = true;

-- Function to get latest prices for a market
CREATE OR REPLACE FUNCTION get_market_prices(p_condition_id VARCHAR)
RETURNS TABLE (
    outcome VARCHAR,
    mid_price DECIMAL,
    best_bid DECIMAL,
    best_ask DECIMAL,
    spread DECIMAL,
    updated_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        o.outcome,
        o.mid_price,
        o.best_bid,
        o.best_ask,
        o.spread,
        o.time as updated_at
    FROM orderbooks o
    JOIN markets m ON m.id = o.market_id
    WHERE m.condition_id = p_condition_id
    AND o.time > NOW() - INTERVAL '5 minutes'
    ORDER BY o.time DESC
    LIMIT 2;
END;
$$ LANGUAGE plpgsql;
