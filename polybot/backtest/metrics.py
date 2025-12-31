"""
Performance metrics calculation for backtests.

Metrics:
- Total return ($ and %)
- Sharpe ratio
- Sortino ratio
- Max drawdown
- Win rate
- Profit factor
"""

from dataclasses import dataclass
from decimal import Decimal
import math


@dataclass
class BacktestMetrics:
    """Backtest performance metrics."""

    initial_balance: float
    final_balance: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration_hours: float


def calculate_metrics(
    trades: list[dict],
    equity_curve: list[float],
    initial_balance: float = 10000.0,
    risk_free_rate: float = 0.05,
) -> BacktestMetrics:
    """
    Calculate comprehensive backtest metrics.

    Args:
        trades: List of trade dictionaries with keys:
            - pnl: Profit/loss amount
            - return_pct: Return percentage
            - duration_hours: Trade duration
        equity_curve: List of portfolio values over time
        initial_balance: Starting capital
        risk_free_rate: Annual risk-free rate for Sharpe calculation

    Returns:
        BacktestMetrics with all calculated values
    """
    if not equity_curve:
        equity_curve = [initial_balance]

    final_balance = equity_curve[-1]
    total_return = final_balance - initial_balance
    total_return_pct = (total_return / initial_balance) * 100 if initial_balance > 0 else 0

    # Separate winning and losing trades
    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0) < 0]

    total_trades = len(trades)
    num_winners = len(winning_trades)
    num_losers = len(losing_trades)

    win_rate = (num_winners / total_trades * 100) if total_trades > 0 else 0

    # Average win/loss
    avg_win = (
        sum(t.get("pnl", 0) for t in winning_trades) / num_winners
        if num_winners > 0
        else 0
    )
    avg_loss = (
        abs(sum(t.get("pnl", 0) for t in losing_trades)) / num_losers
        if num_losers > 0
        else 0
    )

    # Largest win/loss
    largest_win = max((t.get("pnl", 0) for t in trades), default=0)
    largest_loss = min((t.get("pnl", 0) for t in trades), default=0)

    # Profit factor
    gross_profit = sum(t.get("pnl", 0) for t in winning_trades)
    gross_loss = abs(sum(t.get("pnl", 0) for t in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    max_drawdown, max_drawdown_pct = _calculate_max_drawdown(equity_curve)

    # Sharpe ratio
    sharpe_ratio = _calculate_sharpe_ratio(
        equity_curve, risk_free_rate, periods_per_year=365 * 24
    )

    # Sortino ratio
    sortino_ratio = _calculate_sortino_ratio(
        trades, risk_free_rate, periods_per_year=365 * 24
    )

    # Average trade duration
    durations = [t.get("duration_hours", 0) for t in trades if t.get("duration_hours")]
    avg_duration = sum(durations) / len(durations) if durations else 0

    return BacktestMetrics(
        initial_balance=initial_balance,
        final_balance=final_balance,
        total_return=total_return,
        total_return_pct=total_return_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        total_trades=total_trades,
        winning_trades=num_winners,
        losing_trades=num_losers,
        win_rate=win_rate,
        profit_factor=profit_factor if profit_factor != float("inf") else 999.99,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        avg_trade_duration_hours=avg_duration,
    )


def _calculate_max_drawdown(equity_curve: list[float]) -> tuple[float, float]:
    """Calculate maximum drawdown in $ and %."""
    if not equity_curve or len(equity_curve) < 2:
        return 0.0, 0.0

    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    peak = equity_curve[0]

    for value in equity_curve:
        if value > peak:
            peak = value

        drawdown = peak - value
        drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0

        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_pct = drawdown_pct

    return max_drawdown, max_drawdown_pct


def _calculate_sharpe_ratio(
    equity_curve: list[float],
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
    """
    if len(equity_curve) < 2:
        return 0.0

    # Calculate returns
    returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i - 1] > 0:
            ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            returns.append(ret)

    if not returns:
        return 0.0

    # Mean return
    mean_return = sum(returns) / len(returns)

    # Standard deviation
    if len(returns) < 2:
        return 0.0

    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0

    if std_dev == 0:
        return 0.0

    # Annualize
    annualized_return = mean_return * periods_per_year
    annualized_std = std_dev * math.sqrt(periods_per_year)

    sharpe = (annualized_return - risk_free_rate) / annualized_std

    return round(sharpe, 3)


def _calculate_sortino_ratio(
    trades: list[dict],
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).

    Sortino = (Mean Return - Risk Free Rate) / Downside Deviation
    """
    if not trades:
        return 0.0

    returns = [t.get("return_pct", 0) / 100 for t in trades]

    if not returns:
        return 0.0

    mean_return = sum(returns) / len(returns)

    # Downside deviation (only negative returns)
    negative_returns = [r for r in returns if r < 0]

    if not negative_returns:
        return float("inf")

    downside_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns)
    downside_std = math.sqrt(downside_variance) if downside_variance > 0 else 0

    if downside_std == 0:
        return float("inf")

    # Annualize
    annualized_return = mean_return * periods_per_year
    annualized_downside = downside_std * math.sqrt(periods_per_year)

    sortino = (annualized_return - risk_free_rate) / annualized_downside

    return round(min(sortino, 999.99), 3)
