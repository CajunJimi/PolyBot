"""
Data validation before storage.

CRITICAL: All data MUST pass validation before storage.
Invalid data is SKIPPED, never stored with fallback values.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity level for validation issues."""
    ERROR = "error"      # Data rejected
    WARNING = "warning"  # Data accepted with warning
    INFO = "info"        # Informational only


@dataclass
class ValidationError:
    """A single validation error or warning."""
    field: str
    message: str
    severity: ValidationSeverity
    value: Any = None


@dataclass
class ValidationResult:
    """Result of validating a data record."""
    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: int = 0

    @property
    def error_summary(self) -> str:
        """Get summary of errors."""
        error_msgs = [e.message for e in self.errors if e.severity == ValidationSeverity.ERROR]
        return "; ".join(error_msgs) if error_msgs else "Valid"


class DataValidator:
    """
    Validates all incoming data before storage.

    Rules:
    - Prices must be 0-1 range
    - Bid must be <= Ask
    - Required fields must be present
    - Volumes/sizes must be >= 0
    - No synthetic/fake data allowed
    """

    # Price bounds
    MIN_PRICE = 0.0
    MAX_PRICE = 1.0

    # Anomaly detection thresholds
    MAX_PRICE_JUMP = 0.5  # 50% price change is suspicious
    MAX_SPREAD = 0.5      # 50% spread is suspicious

    def __init__(self):
        """Initialize validator with stats tracking."""
        self.stats = {
            "total_validated": 0,
            "total_rejected": 0,
            "markets_validated": 0,
            "markets_rejected": 0,
            "orderbooks_validated": 0,
            "orderbooks_rejected": 0,
            "trades_validated": 0,
            "trades_rejected": 0,
        }

    def validate_market(self, data: dict[str, Any]) -> ValidationResult:
        """
        Validate market data from Gamma API.

        Required fields:
        - condition_id or conditionId
        - slug

        Returns:
            ValidationResult with is_valid=True if data is acceptable
        """
        self.stats["total_validated"] += 1
        self.stats["markets_validated"] += 1
        errors = []

        # Required fields
        condition_id = data.get("condition_id") or data.get("conditionId")
        if not condition_id:
            errors.append(ValidationError(
                field="condition_id",
                message="Missing condition_id",
                severity=ValidationSeverity.ERROR
            ))

        slug = data.get("slug")
        if not slug:
            errors.append(ValidationError(
                field="slug",
                message="Missing slug",
                severity=ValidationSeverity.ERROR
            ))

        # Validate numeric fields if present
        volume = data.get("volume24hr") or data.get("volume_24h")
        if volume is not None:
            try:
                vol = float(volume)
                if vol < 0:
                    errors.append(ValidationError(
                        field="volume_24h",
                        message=f"Negative volume: {vol}",
                        severity=ValidationSeverity.ERROR,
                        value=vol
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    field="volume_24h",
                    message=f"Invalid volume format: {volume}",
                    severity=ValidationSeverity.WARNING,
                    value=volume
                ))

        liquidity = data.get("liquidity")
        if liquidity is not None:
            try:
                liq = float(liquidity)
                if liq < 0:
                    errors.append(ValidationError(
                        field="liquidity",
                        message=f"Negative liquidity: {liq}",
                        severity=ValidationSeverity.ERROR,
                        value=liq
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    field="liquidity",
                    message=f"Invalid liquidity format: {liquidity}",
                    severity=ValidationSeverity.WARNING,
                    value=liquidity
                ))

        # Count warnings
        warnings = sum(1 for e in errors if e.severity == ValidationSeverity.WARNING)
        has_errors = any(e.severity == ValidationSeverity.ERROR for e in errors)

        if has_errors:
            self.stats["total_rejected"] += 1
            self.stats["markets_rejected"] += 1

        return ValidationResult(
            is_valid=not has_errors,
            errors=errors,
            warnings=warnings
        )

    def validate_orderbook(
        self,
        data: dict[str, Any],
        market_id: str | None = None
    ) -> ValidationResult:
        """
        Validate orderbook data from CLOB API.

        Required fields:
        - best_bid
        - best_ask

        Rules:
        - Prices must be 0-1
        - Bid must be <= Ask
        - Spread must be reasonable

        Returns:
            ValidationResult with is_valid=True if data is acceptable
        """
        self.stats["total_validated"] += 1
        self.stats["orderbooks_validated"] += 1
        errors = []

        # Required fields
        best_bid = data.get("best_bid")
        best_ask = data.get("best_ask")

        if best_bid is None:
            errors.append(ValidationError(
                field="best_bid",
                message="Missing best_bid",
                severity=ValidationSeverity.ERROR
            ))
        else:
            try:
                bid = float(best_bid)

                # Price range validation
                if bid < self.MIN_PRICE or bid > self.MAX_PRICE:
                    errors.append(ValidationError(
                        field="best_bid",
                        message=f"Bid price out of range [0,1]: {bid}",
                        severity=ValidationSeverity.ERROR,
                        value=bid
                    ))

                # Zero bid is suspicious but allowed with warning
                if bid == 0:
                    errors.append(ValidationError(
                        field="best_bid",
                        message="Zero bid price",
                        severity=ValidationSeverity.WARNING,
                        value=bid
                    ))

            except (ValueError, TypeError):
                errors.append(ValidationError(
                    field="best_bid",
                    message=f"Invalid bid format: {best_bid}",
                    severity=ValidationSeverity.ERROR,
                    value=best_bid
                ))
                bid = None

        if best_ask is None:
            errors.append(ValidationError(
                field="best_ask",
                message="Missing best_ask",
                severity=ValidationSeverity.ERROR
            ))
        else:
            try:
                ask = float(best_ask)

                # Price range validation
                if ask < self.MIN_PRICE or ask > self.MAX_PRICE:
                    errors.append(ValidationError(
                        field="best_ask",
                        message=f"Ask price out of range [0,1]: {ask}",
                        severity=ValidationSeverity.ERROR,
                        value=ask
                    ))

            except (ValueError, TypeError):
                errors.append(ValidationError(
                    field="best_ask",
                    message=f"Invalid ask format: {best_ask}",
                    severity=ValidationSeverity.ERROR,
                    value=best_ask
                ))
                ask = None

        # Cross-field validation
        if best_bid is not None and best_ask is not None:
            try:
                bid = float(best_bid)
                ask = float(best_ask)

                # Bid must be <= Ask (crossed book is invalid)
                if bid > ask:
                    errors.append(ValidationError(
                        field="spread",
                        message=f"Crossed book: bid ({bid}) > ask ({ask})",
                        severity=ValidationSeverity.ERROR,
                        value={"bid": bid, "ask": ask}
                    ))

                # Large spread is suspicious
                spread = ask - bid
                if spread > self.MAX_SPREAD:
                    errors.append(ValidationError(
                        field="spread",
                        message=f"Suspicious spread: {spread:.4f} ({spread*100:.1f}%)",
                        severity=ValidationSeverity.WARNING,
                        value=spread
                    ))

            except (ValueError, TypeError):
                pass

        # Validate mid_price if provided
        mid_price = data.get("mid_price")
        if mid_price is not None:
            try:
                mid = float(mid_price)
                if mid < self.MIN_PRICE or mid > self.MAX_PRICE:
                    errors.append(ValidationError(
                        field="mid_price",
                        message=f"Mid price out of range [0,1]: {mid}",
                        severity=ValidationSeverity.WARNING,
                        value=mid
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    field="mid_price",
                    message=f"Invalid mid_price format: {mid_price}",
                    severity=ValidationSeverity.WARNING,
                    value=mid_price
                ))

        # Count warnings
        warnings = sum(1 for e in errors if e.severity == ValidationSeverity.WARNING)
        has_errors = any(e.severity == ValidationSeverity.ERROR for e in errors)

        if has_errors:
            self.stats["total_rejected"] += 1
            self.stats["orderbooks_rejected"] += 1

        return ValidationResult(
            is_valid=not has_errors,
            errors=errors,
            warnings=warnings
        )

    def validate_trade(self, data: dict[str, Any]) -> ValidationResult:
        """
        Validate trade data from Data API.

        Required fields:
        - transactionHash or trade_id
        - price
        - size

        Returns:
            ValidationResult with is_valid=True if data is acceptable
        """
        self.stats["total_validated"] += 1
        self.stats["trades_validated"] += 1
        errors = []

        # Required fields
        trade_id = data.get("transactionHash") or data.get("trade_id")
        if not trade_id:
            errors.append(ValidationError(
                field="trade_id",
                message="Missing trade_id/transactionHash",
                severity=ValidationSeverity.ERROR
            ))

        price = data.get("price")
        if price is None:
            errors.append(ValidationError(
                field="price",
                message="Missing price",
                severity=ValidationSeverity.ERROR
            ))
        else:
            try:
                p = float(price)
                if p < self.MIN_PRICE or p > self.MAX_PRICE:
                    errors.append(ValidationError(
                        field="price",
                        message=f"Trade price out of range [0,1]: {p}",
                        severity=ValidationSeverity.ERROR,
                        value=p
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    field="price",
                    message=f"Invalid price format: {price}",
                    severity=ValidationSeverity.ERROR,
                    value=price
                ))

        size = data.get("size")
        if size is None:
            errors.append(ValidationError(
                field="size",
                message="Missing size",
                severity=ValidationSeverity.ERROR
            ))
        else:
            try:
                s = float(size)
                if s < 0:
                    errors.append(ValidationError(
                        field="size",
                        message=f"Negative trade size: {s}",
                        severity=ValidationSeverity.ERROR,
                        value=s
                    ))
                if s == 0:
                    errors.append(ValidationError(
                        field="size",
                        message="Zero trade size",
                        severity=ValidationSeverity.WARNING,
                        value=s
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    field="size",
                    message=f"Invalid size format: {size}",
                    severity=ValidationSeverity.ERROR,
                    value=size
                ))

        # Validate side if present
        side = data.get("side")
        if side:
            if str(side).upper() not in ("BUY", "SELL"):
                errors.append(ValidationError(
                    field="side",
                    message=f"Invalid side: {side}",
                    severity=ValidationSeverity.WARNING,
                    value=side
                ))

        # Count warnings
        warnings = sum(1 for e in errors if e.severity == ValidationSeverity.WARNING)
        has_errors = any(e.severity == ValidationSeverity.ERROR for e in errors)

        if has_errors:
            self.stats["total_rejected"] += 1
            self.stats["trades_rejected"] += 1

        return ValidationResult(
            is_valid=not has_errors,
            errors=errors,
            warnings=warnings
        )

    def get_stats(self) -> dict[str, Any]:
        """Get validation statistics."""
        return {
            **self.stats,
            "rejection_rate": (
                self.stats["total_rejected"] / max(self.stats["total_validated"], 1) * 100
            ),
        }

    def reset_stats(self):
        """Reset statistics."""
        for key in self.stats:
            self.stats[key] = 0


# Singleton instance
_validator: DataValidator | None = None


def get_validator() -> DataValidator:
    """Get global validator instance."""
    global _validator
    if _validator is None:
        _validator = DataValidator()
    return _validator
