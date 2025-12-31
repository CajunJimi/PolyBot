"""
Paper Trading - Simulated Balance Tracking.

Tracks virtual balances for each strategy during testing.
Persists to JSON file so state survives restarts.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# State file location
STATE_FILE = Path.home() / ".polybot" / "paper_trading.json"


@dataclass
class StrategyAccount:
    """Simulated account for a strategy."""

    name: str
    starting_balance: float = 1000.0
    current_balance: float = 1000.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    peak_balance: float = 1000.0
    max_drawdown: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def pnl_percent(self) -> float:
        """Calculate P/L percentage."""
        if self.starting_balance == 0:
            return 0.0
        return ((self.current_balance - self.starting_balance) / self.starting_balance) * 100

    def record_trade(self, pnl: float):
        """Record a trade result."""
        self.total_trades += 1
        self.total_pnl += pnl
        self.current_balance += pnl

        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Track peak and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        if self.peak_balance > 0:
            drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "balance": self.current_balance,
            "starting_balance": self.starting_balance,
            "pnl": self.total_pnl,
            "pnl_percent": self.pnl_percent,
            "trades": self.total_trades,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
        }


class PaperTrader:
    """
    Manages paper trading accounts for all strategies.

    Each strategy gets a simulated balance to track performance.
    State persists to JSON file.
    """

    def __init__(self, starting_balance: float = 1000.0):
        """Initialize paper trader."""
        self.starting_balance = starting_balance
        self.accounts: dict[str, StrategyAccount] = {}
        self.trade_history: list[dict] = []
        self.created_at = datetime.utcnow()

        # Ensure state directory exists
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load existing state
        self._load_state()

        if self.accounts:
            total = sum(a.current_balance for a in self.accounts.values())
            logger.info(
                f"Paper Trader restored: {len(self.accounts)} accounts, "
                f"${total:,.2f} total, {len(self.trade_history)} trades"
            )
        else:
            logger.info(f"Paper Trader initialized with ${starting_balance:,.2f} per strategy")

    def _load_state(self):
        """Load state from JSON file."""
        try:
            if STATE_FILE.exists():
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)

                for name, acc_data in data.get("accounts", {}).items():
                    self.accounts[name] = StrategyAccount(
                        name=name,
                        starting_balance=self.starting_balance,
                        current_balance=acc_data.get("balance", self.starting_balance),
                        total_trades=acc_data.get("trades", 0),
                        winning_trades=int(acc_data.get("trades", 0) * acc_data.get("win_rate", 0)),
                        losing_trades=int(acc_data.get("trades", 0) * (1 - acc_data.get("win_rate", 0))),
                        total_pnl=acc_data.get("pnl", 0.0),
                        max_drawdown=acc_data.get("max_drawdown", 0.0),
                    )

                self.trade_history = data.get("trade_history", [])
                logger.info(f"Loaded paper trading state from {STATE_FILE}")

        except Exception as e:
            logger.warning(f"Could not load paper trading state: {e}")

    def get_account(self, strategy_name: str) -> StrategyAccount:
        """Get or create account for a strategy."""
        if strategy_name not in self.accounts:
            self.accounts[strategy_name] = StrategyAccount(
                name=strategy_name,
                starting_balance=self.starting_balance,
                current_balance=self.starting_balance,
            )
            logger.info(f"Created paper account for {strategy_name}: ${self.starting_balance:,.2f}")

        return self.accounts[strategy_name]

    def execute_trade(
        self,
        strategy: str,
        signal: Any,
        size_usd: float = 50.0,
        exit_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Execute a paper trade based on a signal.

        Args:
            strategy: Strategy name
            signal: Signal object with entry_price, action, confidence
            size_usd: Trade size in USD
            exit_price: Exit price (or use current price from signal)

        Returns:
            Trade result dict
        """
        try:
            account = self.get_account(strategy)

            # Check balance
            if account.current_balance < size_usd:
                return {"success": False, "error": "Insufficient balance"}

            # Get signal details
            entry_price = getattr(signal, "entry_price", 0.5) or 0.5
            action = getattr(signal, "action", "buy_yes")
            if hasattr(action, "value"):
                action = action.value
            confidence = getattr(signal, "confidence", 0.5) or 0.5

            # Use provided exit price or current price
            actual_exit = exit_price if exit_price is not None else getattr(signal, "current_price", entry_price)

            # Calculate P/L based on price movement
            if "yes" in str(action).lower():
                # Bought YES - profit if price goes UP
                price_change = actual_exit - entry_price
            else:
                # Bought NO - profit if YES price goes DOWN
                price_change = entry_price - actual_exit

            # P/L = price change * (size / entry_price)
            pnl = price_change * (size_usd / entry_price) if entry_price > 0 else 0

            # Record trade
            account.record_trade(pnl)

            # Record in history
            market_slug = getattr(signal, "market_slug", "unknown") or "unknown"
            trade_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "strategy": strategy,
                "market": market_slug[:50],
                "action": str(action),
                "entry_price": entry_price,
                "exit_price": actual_exit,
                "size": size_usd,
                "pnl": round(pnl, 2),
                "balance_after": round(account.current_balance, 2),
                "confidence": round(confidence, 2),
                "result": "WIN" if pnl > 0 else "LOSS",
            }
            self.trade_history.append(trade_record)

            # Keep only last 500 trades
            if len(self.trade_history) > 500:
                self.trade_history = self.trade_history[-500:]

            # Save state
            self.save_state()

            emoji = "+" if pnl >= 0 else ""
            logger.info(
                f"[{strategy}] Paper Trade: {action} @ {entry_price:.3f} -> {actual_exit:.3f} | "
                f"P/L: ${emoji}{pnl:.2f} | Balance: ${account.current_balance:,.2f}"
            )

            return {
                "success": True,
                "action": str(action),
                "entry_price": entry_price,
                "exit_price": actual_exit,
                "size": size_usd,
                "pnl": pnl,
                "new_balance": account.current_balance,
            }

        except Exception as e:
            logger.error(f"Paper trade error: {e}")
            return {"success": False, "error": str(e)}

    def record_pnl(self, strategy_name: str, pnl: float):
        """Record a P/L directly."""
        account = self.get_account(strategy_name)
        account.record_trade(pnl)

        emoji = "+" if pnl >= 0 else ""
        logger.info(
            f"[{strategy_name}] P/L: ${emoji}{pnl:.2f} | "
            f"Balance: ${account.current_balance:,.2f} ({account.pnl_percent:+.1f}%)"
        )

        self.save_state()

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all accounts."""
        total_balance = sum(a.current_balance for a in self.accounts.values())
        total_pnl = sum(a.total_pnl for a in self.accounts.values())
        total_trades = sum(a.total_trades for a in self.accounts.values())

        return {
            "total_balance": total_balance,
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "starting_balance": self.starting_balance,
            "accounts": {name: acc.to_dict() for name, acc in self.accounts.items()},
        }

    def save_state(self):
        """Save state to JSON file."""
        try:
            summary = self.get_summary()
            summary["updated_at"] = datetime.utcnow().isoformat()
            summary["trade_history"] = self.trade_history

            with open(STATE_FILE, "w") as f:
                json.dump(summary, f, indent=2)

        except Exception as e:
            logger.debug(f"Could not save paper trading state: {e}")

    @classmethod
    def load_state_file(cls) -> dict[str, Any]:
        """Load state from file (for dashboard)."""
        try:
            if STATE_FILE.exists():
                with open(STATE_FILE, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Could not load state file: {e}")

        return {
            "total_balance": 0,
            "total_pnl": 0,
            "total_trades": 0,
            "accounts": {},
            "trade_history": [],
        }

    def print_status(self):
        """Print status of all accounts."""
        logger.info("\n" + "=" * 60)
        logger.info("PAPER TRADING STATUS")
        logger.info("=" * 60)

        for name, account in self.accounts.items():
            logger.info(
                f"  {name:20s} | "
                f"Balance: ${account.current_balance:>10,.2f} | "
                f"P/L: ${account.total_pnl:>+8,.2f} ({account.pnl_percent:>+5.1f}%) | "
                f"Trades: {account.total_trades:>3} | "
                f"Win: {account.win_rate*100:>5.1f}%"
            )

        summary = self.get_summary()
        logger.info("-" * 60)
        logger.info(
            f"  {'TOTAL':20s} | "
            f"Balance: ${summary['total_balance']:>10,.2f} | "
            f"P/L: ${summary['total_pnl']:>+8,.2f}"
        )
        logger.info("=" * 60 + "\n")


# Singleton
_paper_trader: PaperTrader | None = None


def get_paper_trader(starting_balance: float = 1000.0) -> PaperTrader:
    """Get singleton paper trader."""
    global _paper_trader
    if _paper_trader is None:
        _paper_trader = PaperTrader(starting_balance)
    return _paper_trader
