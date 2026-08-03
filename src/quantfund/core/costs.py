"""Execution cost model: spread crossing, slippage, commissions, regulatory fees.

Invariant enforced by tests: a simulated fill NEVER executes at a better price
than the touch (buy at >= ask, sell at <= bid, given a valid two-sided quote),
and total friction versus mid is strictly positive whenever the quote has a
positive spread or fees apply. A zero-cost fill is a bug.
"""
from __future__ import annotations

from dataclasses import dataclass

from quantfund.core.instruments import AssetClass, Instrument
from quantfund.core.orders import OrderSide
from quantfund.core.snapshot import Quote


@dataclass(frozen=True)
class CostConfig:
    # Equities: Alpaca is commission-free, but regulatory fees exist on sells
    # and market impact is real. Defaults are deliberately conservative.
    equity_slippage_bps: float = 2.0        # extra impact beyond crossing the spread
    sec_fee_rate: float = 0.0000278         # SEC section 31 fee on sell notional
    taf_per_share: float = 0.000166         # FINRA TAF per share sold (capped at $8.30)
    taf_cap: float = 8.30

    # Options: fees per contract (OCC clearing + exchange + regulatory, rounded up)
    option_fee_per_contract: float = 0.65
    # Fraction of half-spread we assume as price improvement when crossing.
    # 0.0 = pay the full touch (most conservative). Never > 0.5 (that'd be mid).
    option_price_improvement: float = 0.0
    equity_price_improvement: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.option_price_improvement <= 0.5):
            raise ValueError("option_price_improvement must be in [0, 0.5]")
        if not (0.0 <= self.equity_price_improvement <= 0.5):
            raise ValueError("equity_price_improvement must be in [0, 0.5]")


@dataclass(frozen=True)
class FillCosts:
    price: float          # per-share execution price (spread + slippage included)
    commission: float     # >= 0
    fees: float           # >= 0
    slippage_cost: float  # dollar friction vs mid fill, >= 0


class CostModel:
    """Computes realistic execution prices and costs from a live/synthetic quote."""

    def __init__(self, config: CostConfig | None = None):
        self.config = config or CostConfig()

    def execution_price(self, instrument: Instrument, side: OrderSide, quote: Quote) -> float:
        """Price actually paid: cross the spread toward the touch, plus impact.

        Requires a valid two-sided quote (bid > 0 and ask > 0); raises otherwise
        so callers can't silently fill off garbage data.
        """
        if quote.bid <= 0 or quote.ask <= 0:
            raise ValueError(
                f"cannot price {instrument.key}: need two-sided quote, got "
                f"bid={quote.bid} ask={quote.ask}"
            )
        mid = quote.mid
        half_spread = 0.5 * quote.spread
        if instrument.asset_class == AssetClass.OPTION:
            improvement = self.config.option_price_improvement
            slip = 0.0  # options friction is dominated by the spread itself
        else:
            improvement = self.config.equity_price_improvement
            slip = mid * (self.config.equity_slippage_bps / 10_000.0)
        crossed = half_spread * (1.0 - improvement)
        if side == OrderSide.BUY:
            return mid + crossed + slip
        return max(0.0, mid - crossed - slip)

    def commission_and_fees(
        self, instrument: Instrument, side: OrderSide, qty: float, price: float
    ) -> tuple[float, float]:
        """Returns (commission, regulatory_fees), both >= 0."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        if instrument.asset_class == AssetClass.OPTION:
            commission = self.config.option_fee_per_contract * qty
            return commission, 0.0
        commission = 0.0
        fees = 0.0
        if side == OrderSide.SELL:
            notional = price * qty
            fees += self.config.sec_fee_rate * notional
            fees += min(self.config.taf_per_share * qty, self.config.taf_cap)
        return commission, fees

    def simulate_fill(
        self, instrument: Instrument, side: OrderSide, qty: float, quote: Quote
    ) -> FillCosts:
        """Full fill simulation for one leg. Never returns a zero-friction fill
        when the quote has a positive spread."""
        price = self.execution_price(instrument, side, quote)
        commission, fees = self.commission_and_fees(instrument, side, qty, price)
        mid = quote.mid
        slippage_cost = abs(price - mid) * qty * instrument.multiplier
        return FillCosts(price=price, commission=commission, fees=fees,
                         slippage_cost=slippage_cost)
