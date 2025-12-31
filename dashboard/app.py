"""
PolyBot Streamlit Dashboard.

Simple, Python-only dashboard for:
1. Data Health - Collection status and metrics
2. Market Explorer - Browse markets and prices
3. Backtest Results - Strategy performance
4. Signals - Trading signals
"""

import asyncio
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select, func, text

# Page configuration
st.set_page_config(
    page_title="PolyBot Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Database connection (synchronous wrapper for Streamlit)
def get_db_connection():
    """Get synchronous database connection for Streamlit."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # Use synchronous URL for Streamlit
    db_url = "postgresql://polybot:polybot_dev_password@localhost:5432/polybot"
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    return Session()


def get_collection_stats():
    """Get data collection statistics."""
    try:
        session = get_db_connection()

        # Count markets
        markets_count = session.execute(
            text("SELECT COUNT(*) FROM markets WHERE active = true")
        ).scalar()

        # Orderbooks in last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        orderbooks_1h = session.execute(
            text("SELECT COUNT(*) FROM orderbooks WHERE time > :time"),
            {"time": one_hour_ago},
        ).scalar()

        # Trades in last hour
        trades_1h = session.execute(
            text("SELECT COUNT(*) FROM trades WHERE time > :time"),
            {"time": one_hour_ago},
        ).scalar()

        # Latest orderbook timestamp
        latest_ob = session.execute(
            text("SELECT MAX(time) FROM orderbooks")
        ).scalar()

        # Health metrics
        health = session.execute(
            text("""
                SELECT component,
                       SUM(successes) as successes,
                       SUM(failures) as failures,
                       AVG(avg_latency_ms) as avg_latency
                FROM collection_health
                WHERE timestamp > :time
                GROUP BY component
            """),
            {"time": one_hour_ago},
        ).fetchall()

        session.close()

        return {
            "markets_count": markets_count or 0,
            "orderbooks_1h": orderbooks_1h or 0,
            "trades_1h": trades_1h or 0,
            "latest_update": latest_ob,
            "health": {h[0]: {"successes": h[1], "failures": h[2], "latency": h[3]} for h in health},
        }
    except Exception as e:
        st.error(f"Database error: {e}")
        return None


def get_markets_df():
    """Get markets as DataFrame."""
    try:
        session = get_db_connection()
        result = session.execute(
            text("""
                SELECT slug, question, category, volume_24h, liquidity, active, updated_at
                FROM markets
                WHERE active = true
                ORDER BY volume_24h DESC
                LIMIT 100
            """)
        ).fetchall()
        session.close()

        return pd.DataFrame(
            result,
            columns=["Slug", "Question", "Category", "Volume 24h", "Liquidity", "Active", "Updated"],
        )
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


def get_orderbook_history(market_slug: str, hours: int = 24):
    """Get orderbook history for a market."""
    try:
        session = get_db_connection()
        time_ago = datetime.utcnow() - timedelta(hours=hours)

        result = session.execute(
            text("""
                SELECT o.time, o.outcome, o.mid_price, o.spread
                FROM orderbooks o
                JOIN markets m ON m.id = o.market_id
                WHERE m.slug = :slug AND o.time > :time
                ORDER BY o.time
            """),
            {"slug": market_slug, "time": time_ago},
        ).fetchall()
        session.close()

        return pd.DataFrame(result, columns=["Time", "Outcome", "Mid Price", "Spread"])
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


def get_backtest_results():
    """Get backtest results."""
    try:
        session = get_db_connection()
        result = session.execute(
            text("""
                SELECT strategy_name, start_date, end_date,
                       total_return_pct, sharpe_ratio, win_rate,
                       total_trades, max_drawdown_pct, created_at
                FROM backtest_results
                ORDER BY created_at DESC
                LIMIT 20
            """)
        ).fetchall()
        session.close()

        return pd.DataFrame(
            result,
            columns=[
                "Strategy", "Start", "End", "Return %", "Sharpe",
                "Win Rate", "Trades", "Max DD %", "Run Date"
            ],
        )
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


def get_signals():
    """Get recent signals."""
    try:
        session = get_db_connection()
        result = session.execute(
            text("""
                SELECT strategy_name, market_slug, action, confidence,
                       expected_edge, entry_price, status, created_at
                FROM signals
                ORDER BY created_at DESC
                LIMIT 50
            """)
        ).fetchall()
        session.close()

        return pd.DataFrame(
            result,
            columns=[
                "Strategy", "Market", "Action", "Confidence",
                "Edge", "Entry Price", "Status", "Created"
            ],
        )
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


# Sidebar
st.sidebar.title("PolyBot")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Data Health", "Market Explorer", "Backtest Results", "Signals"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")

stats = get_collection_stats()
if stats:
    st.sidebar.metric("Active Markets", stats["markets_count"])
    st.sidebar.metric("Orderbooks/hr", stats["orderbooks_1h"])
    st.sidebar.metric("Trades/hr", stats["trades_1h"])


# Main content
if page == "Data Health":
    st.title("📊 Data Health")

    if stats:
        # Status cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Active Markets",
                stats["markets_count"],
                help="Number of active markets being tracked",
            )

        with col2:
            st.metric(
                "Orderbooks (1h)",
                f"{stats['orderbooks_1h']:,}",
                help="Orderbook snapshots collected in last hour",
            )

        with col3:
            st.metric(
                "Trades (1h)",
                f"{stats['trades_1h']:,}",
                help="Trades collected in last hour",
            )

        with col4:
            if stats["latest_update"]:
                age = datetime.utcnow() - stats["latest_update"]
                age_str = f"{age.seconds}s ago"
                st.metric("Last Update", age_str)
            else:
                st.metric("Last Update", "No data")

        st.markdown("---")

        # Health metrics
        st.subheader("Collection Health")

        if stats["health"]:
            health_data = []
            for component, metrics in stats["health"].items():
                total = (metrics["successes"] or 0) + (metrics["failures"] or 0)
                success_rate = (
                    (metrics["successes"] or 0) / total * 100 if total > 0 else 0
                )
                health_data.append({
                    "Component": component,
                    "Success Rate": f"{success_rate:.1f}%",
                    "Successes": metrics["successes"] or 0,
                    "Failures": metrics["failures"] or 0,
                    "Avg Latency (ms)": f"{metrics['latency']:.0f}" if metrics["latency"] else "N/A",
                })

            st.dataframe(pd.DataFrame(health_data), use_container_width=True)
        else:
            st.info("No health data available yet. Run the collector first.")

        st.markdown("---")

        # Instructions
        st.subheader("Getting Started")
        st.code("""
# Start PostgreSQL + Redis
docker-compose up -d

# Run the data collector
python scripts/run_collector.py

# Verify data is being collected
docker exec -it polybot-postgres psql -U polybot -c "SELECT COUNT(*) FROM orderbooks;"
        """, language="bash")

    else:
        st.warning("Could not connect to database. Make sure PostgreSQL is running.")
        st.code("docker-compose up -d", language="bash")


elif page == "Market Explorer":
    st.title("🔍 Market Explorer")

    markets_df = get_markets_df()

    if not markets_df.empty:
        # Filters
        col1, col2 = st.columns(2)

        with col1:
            search = st.text_input("Search markets", "")

        with col2:
            categories = ["All"] + sorted(markets_df["Category"].dropna().unique().tolist())
            category = st.selectbox("Category", categories)

        # Apply filters
        filtered_df = markets_df
        if search:
            filtered_df = filtered_df[
                filtered_df["Slug"].str.contains(search, case=False, na=False)
                | filtered_df["Question"].str.contains(search, case=False, na=False)
            ]
        if category != "All":
            filtered_df = filtered_df[filtered_df["Category"] == category]

        st.dataframe(
            filtered_df,
            use_container_width=True,
            column_config={
                "Volume 24h": st.column_config.NumberColumn(format="$%.0f"),
                "Liquidity": st.column_config.NumberColumn(format="$%.0f"),
            },
        )

        st.markdown("---")

        # Market detail
        st.subheader("Price Chart")
        selected_slug = st.selectbox(
            "Select market",
            filtered_df["Slug"].tolist() if not filtered_df.empty else [],
        )

        if selected_slug:
            hours = st.slider("Hours of history", 1, 72, 24)
            history_df = get_orderbook_history(selected_slug, hours)

            if not history_df.empty:
                fig = px.line(
                    history_df,
                    x="Time",
                    y="Mid Price",
                    color="Outcome",
                    title=f"Price History: {selected_slug}",
                )
                fig.update_layout(yaxis_tickformat=".2%")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No price history available for this market.")

    else:
        st.info("No markets found. Run the collector to discover markets.")


elif page == "Backtest Results":
    st.title("📈 Backtest Results")

    results_df = get_backtest_results()

    if not results_df.empty:
        st.dataframe(
            results_df,
            use_container_width=True,
            column_config={
                "Return %": st.column_config.NumberColumn(format="%.2f%%"),
                "Sharpe": st.column_config.NumberColumn(format="%.2f"),
                "Win Rate": st.column_config.NumberColumn(format="%.1f%%"),
                "Max DD %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

        st.markdown("---")

        # Summary stats
        st.subheader("Strategy Comparison")

        summary = results_df.groupby("Strategy").agg({
            "Return %": "mean",
            "Sharpe": "mean",
            "Win Rate": "mean",
            "Trades": "sum",
        }).round(2)

        st.dataframe(summary, use_container_width=True)

    else:
        st.info("No backtest results yet. Run a backtest first.")
        st.code("python scripts/run_backtest.py", language="bash")


elif page == "Signals":
    st.title("🎯 Trading Signals")

    signals_df = get_signals()

    if not signals_df.empty:
        # Filter by status
        status_filter = st.multiselect(
            "Filter by status",
            signals_df["Status"].unique().tolist(),
            default=signals_df["Status"].unique().tolist(),
        )

        filtered_signals = signals_df[signals_df["Status"].isin(status_filter)]

        st.dataframe(
            filtered_signals,
            use_container_width=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                ),
                "Edge": st.column_config.NumberColumn(format="%.2f%%"),
                "Entry Price": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        st.markdown("---")

        # Signal summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Signals", len(signals_df))

        with col2:
            pending = len(signals_df[signals_df["Status"] == "pending"])
            st.metric("Pending", pending)

        with col3:
            avg_conf = signals_df["Confidence"].mean()
            st.metric("Avg Confidence", f"{avg_conf:.2f}")

    else:
        st.info("No signals generated yet. Run a strategy scan first.")


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **PolyBot v0.1.0**

    [GitHub](https://github.com/yourusername/polybot)
    """
)
