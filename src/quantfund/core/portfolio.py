"""Unified position/portfolio accounting for equities and options.

Cash, cost basis, realized/unrealized P&L, exposures, and net Greeks are all
tracked here. The same class serves the live paper account mirror, per-sleeve
sub-portfolios, and the backtester — one accounting implementation, tested once.

Conventions (see instruments.py): prices are per-share; notional = price * qty
* multiplier. ``qty`` on a Position is SIGNED (negative = short). ``realized_pnl``
is price-based (excludes commissions/fees); friction is tracked separately in
``total_costs`` and is already reflected in ``cash``, so ``equity()`` is always
net of all costs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from quantfund.core.greeks import Greeks, bs_greeks, implied_vol
from quantfund.core.instruments import AssetClass, Equity, Instrument, Option
from quantfund.core.orders import Fill
from quantfund.core.snapshot import MarketSnapshot


# Below this |delta| a LONG option is a capped-loss ticket: its premium, not
# its delta notional, is what can actually be lost. At or above it the contract
# behaves like leveraged stock (a 0.85-delta call tracks the underlying nearly
# 1:1 and its premium is large), so delta notional is the honest measure.
#
# 0.70, not 0.50. The threshold has to separate STOCK REPLACEMENT from LOTTERY
# TICKET, and an at-the-money option is not stock replacement — it has roughly
# even odds of expiring worthless. At 0.50 the cliff sat exactly where 0DTE
# trades: a 0.49-delta contract measured $4.5k of premium and a 0.51-delta
# contract measured $1.79M of delta notional, so identical trades were
# accepted or rejected on a rounding error in delta.
# This does NOT remove sensitivity control: net dollar delta (risk check 8)
# still counts RAW delta for every option and is capped by
# max_net_delta_pct. The per-underlying and per-sector caps are
# loss-concentration limits, which is what they should be for a capped-loss
# long. The momentum sleeve's 0.80-delta stock-replacement calls stay on
# delta notional, which is the case this threshold exists to catch.
ITM_DELTA_THRESHOLD = 0.70


def option_group_key(inst) -> tuple:
    """(underlying, right, expiration) — the unit within which longs and
    shorts offset each other, matching the naked-short coverage check."""
    return (inst.underlying_symbol, inst.right, inst.expiration)


def option_exposure(qty: float, multiplier: float, delta: Optional[float],
                    underlying_price: Optional[float], premium: Optional[float],
                    fallback: float = 0.0, group_has_short: bool = False) -> float:
    """Directional exposure of one option position, by moneyness.

    SHORT options always use delta notional — their loss is unbounded, so
    nothing about premium bounds them.

    LONG options split on |delta|:
      * |delta| <  ITM_DELTA_THRESHOLD -> signed PREMIUM notional. A 0DTE
        0.33-delta call shows $510k of delta notional against $1,911 of
        premium; it cannot lose more than the premium, so measuring it by
        delta both overstates the risk and crowds every other sleeve out of
        the per-underlying and per-sector budgets.
      * |delta| >= ITM_DELTA_THRESHOLD -> delta notional. A deep-ITM call is
        stock replacement; pricing it at premium would hide real leverage.

    Used by BOTH Portfolio.exposure_by_underlying and the pre-trade risk gate.
    They MUST agree: when the gate sized 0DTE by premium while the book marked
    it by Black-Scholes delta, a 21-lot was authorized as $1,911 of exposure
    and landed as $510,170 (2026-08-07).
    """
    if delta is None or not underlying_price or underlying_price <= 0:
        return fallback
    delta_notional = delta * qty * multiplier * underlying_price
    if qty < 0:
        return delta_notional
    # SPREADS: if this group also holds shorts, measure the long on the SAME
    # basis so the legs offset. Mixing bases is not additive — a defined-risk
    # vertical showed the long leg at premium and its covering short at full
    # delta notional, so the spread read like a large naked short and was
    # rejected. Delta notional is additive and nets correctly across legs.
    if group_has_short:
        return delta_notional
    if abs(delta) >= ITM_DELTA_THRESHOLD:
        return delta_notional
    if premium is None or premium <= 0:
        return fallback
    # keep the DIRECTION (long put is short the underlying), size by premium
    sign = 1.0 if delta >= 0 else -1.0
    return sign * premium * qty * multiplier


def option_theta_per_day(qty: float, multiplier: float,
                        theta: Optional[float],
                        premium: Optional[float]) -> float:
    """Daily decay of one option position, bounded by what can actually decay.

    Black-Scholes theta is an INSTANTANEOUS rate. As time-to-expiry goes to
    zero it explodes: a 21-lot QQQ 0DTE call carrying $1,932 of total premium
    reports -$7,930/day. That rate is never realised — a long option cannot
    decay past zero, so its worst case over a day is the premium itself.

    Same principle as ``option_exposure``: for a capped-loss LONG, the premium
    is the bound. Short options are NOT capped (decay works for you, but the
    position's risk is unbounded elsewhere), so they pass through unmodified.
    """
    if theta is None:
        return 0.0
    raw = theta * qty * multiplier
    if qty <= 0 or premium is None or premium <= 0:
        return raw
    max_decay = premium * qty * multiplier      # cannot lose more than paid
    return -min(abs(raw), abs(max_decay)) if raw < 0 else raw


@dataclass
class Position:
    instrument: Instrument
    qty: float = 0.0                 # signed; shares or contracts
    avg_cost: float = 0.0            # per share
    realized_pnl: float = 0.0        # price-based, cumulative, in dollars
    mark: Optional[float] = None     # per-share mark price
    mark_ts: Optional[datetime] = None
    greeks: Optional[Greeks] = None  # per-share greeks at last mark (options)
    underlying_mark: Optional[float] = None  # last underlying price (options)

    @property
    def is_open(self) -> bool:
        return abs(self.qty) > 1e-12

    @property
    def multiplier(self) -> int:
        return self.instrument.multiplier

    @property
    def market_value(self) -> float:
        """Signed liquidation value at the current mark (0 if never marked)."""
        price = self.mark if self.mark is not None else self.avg_cost
        return price * self.qty * self.multiplier

    @property
    def notional(self) -> float:
        return abs(self.market_value)

    @property
    def unrealized_pnl(self) -> float:
        if not self.is_open or self.mark is None:
            return 0.0
        return (self.mark - self.avg_cost) * self.qty * self.multiplier

    def apply_fill(self, fill: Fill) -> float:
        """Update qty/avg_cost, return realized P&L generated by this fill."""
        signed = fill.side.sign * fill.qty
        realized = 0.0
        if self.qty == 0 or (self.qty > 0) == (signed > 0):
            # opening or adding: weighted-average cost basis
            total = abs(self.qty) + fill.qty
            self.avg_cost = (abs(self.qty) * self.avg_cost + fill.qty * fill.price) / total
            self.qty += signed
        else:
            closing = min(abs(signed), abs(self.qty))
            direction = 1.0 if self.qty > 0 else -1.0
            realized = (fill.price - self.avg_cost) * closing * direction * self.multiplier
            self.realized_pnl += realized
            remaining = abs(signed) - closing
            self.qty += signed
            if remaining > 0:
                # flipped through zero: leftover opens at the fill price
                self.avg_cost = fill.price
            elif abs(self.qty) < 1e-12:
                self.qty = 0.0
                self.avg_cost = 0.0
        return realized


@dataclass(frozen=True)
class PortfolioGreeks:
    """Portfolio-level risk in dollar terms.

    dollar_delta: $ P&L per +100% move of each underlying, summed (equity qty*px
                  counts as delta 1). This equals net directional notional.
    gamma_shares: change in share-equivalent delta per $1 underlying move.
    theta_per_day: $ per calendar day from option decay.
    vega: $ per +1 vol point across all options.
    """

    dollar_delta: float = 0.0
    gamma_shares: float = 0.0
    theta_per_day: float = 0.0
    vega: float = 0.0


@dataclass
class Portfolio:
    cash: float
    starting_equity: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    total_costs: float = 0.0            # cumulative commissions + fees + slippage
    total_commissions_fees: float = 0.0  # cash costs only (already out of cash)
    day_anchor_equity: Optional[float] = None
    peak_equity: float = 0.0

    def __post_init__(self) -> None:
        if self.starting_equity == 0.0:
            self.starting_equity = self.cash
        self.peak_equity = max(self.peak_equity, self.cash)

    # ── trade application ────────────────────────────────────────────────

    def apply_fill(self, fill: Fill) -> float:
        """Apply a fill: cash moves by fill.cash_flow, position updates.
        Returns realized P&L from this fill (price-based)."""
        pos = self.positions.get(fill.instrument.key)
        if pos is None:
            pos = Position(instrument=fill.instrument)
            self.positions[fill.instrument.key] = pos
        realized = pos.apply_fill(fill)
        self.cash += fill.cash_flow
        self.total_commissions_fees += fill.commission + fill.fees
        self.total_costs += fill.total_friction
        # keep the mark current so equity() doesn't jump on the next mark
        pos.mark = fill.price
        pos.mark_ts = fill.ts
        return realized

    # ── marking ──────────────────────────────────────────────────────────

    def mark_position(self, key: str, price: float, ts: datetime,
                      greeks: Optional[Greeks] = None,
                      underlying_price: Optional[float] = None) -> None:
        pos = self.positions.get(key)
        if pos is None:
            return
        pos.mark = price
        pos.mark_ts = ts
        if greeks is not None:
            pos.greeks = greeks
        if underlying_price is not None:
            pos.underlying_mark = underlying_price

    def mark_from_snapshot(self, snapshot: MarketSnapshot, risk_free: float = 0.04) -> None:
        """Mark every open position from a point-in-time snapshot.

        Options are priced off the chain (bid/ask mid for marking; execution uses
        the cost model, never mid). Greeks come from the chain when present,
        else Black-Scholes off chain IV, else IV implied from the quote itself.
        """
        for pos in self.positions.values():
            if not pos.is_open:
                continue
            inst = pos.instrument
            if isinstance(inst, Equity):
                price = snapshot.last_price(inst.symbol)
                if price is not None:
                    self.mark_position(inst.key, price, snapshot.as_of)
            elif isinstance(inst, Option):
                chain = snapshot.get_option_chain(inst.underlying_symbol)
                under_px = snapshot.last_price(inst.underlying_symbol)
                if chain is None:
                    # No chain in view: decay-free fallback — BS off last known IV
                    if under_px is not None and pos.greeks is not None:
                        pass  # keep previous mark; do not invent prices
                    continue
                oq = next(
                    (q for q in chain.quotes if q.instrument.symbol == inst.symbol), None
                )
                if oq is None:
                    continue
                price = oq.quote.mid
                if price <= 0:
                    continue
                under_px = under_px if under_px is not None else chain.underlying_price
                greeks = oq.greeks
                if greeks is None and under_px and under_px > 0:
                    t = inst.years_to_expiration(snapshot.as_of)
                    iv = oq.iv
                    if iv is None:
                        iv = implied_vol(inst.right, price, under_px, inst.strike, t, risk_free)
                    if iv is not None and t > 0:
                        greeks = bs_greeks(inst.right, under_px, inst.strike, t, iv, risk_free)
                self.mark_position(inst.key, price, snapshot.as_of, greeks=greeks,
                                   underlying_price=under_px)
        self.peak_equity = max(self.peak_equity, self.equity())

    # ── valuation & risk ─────────────────────────────────────────────────

    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values() if p.is_open)

    def market_value(self) -> float:
        return sum(p.market_value for p in self.positions.values() if p.is_open)

    def gross_exposure(self) -> float:
        """Sum of |market value|. For options this is PREMIUM, not economic
        exposure — a $1.8k deep-ITM call controls ~$65k of stock. Use
        ``gross_delta_exposure`` to judge how deployed the book actually is;
        this number is what is at risk of going to zero, which is a different
        (and for long options, the more relevant) question."""
        return sum(p.notional for p in self.positions.values() if p.is_open)

    def gross_delta_exposure(self) -> float:
        """Economic exposure: delta-adjusted notional, summed absolutely.

        This is the number that answers "how much of the portfolio is
        working?" — and it is the one the risk gate enforces per-underlying
        and per-sector. Reporting only ``gross_exposure`` made a book with
        ~1x delta exposure look 19% deployed, because option premium is a
        small fraction of the stock an option controls.
        """
        return sum(abs(v) for v in self.exposure_by_underlying().values())

    def net_exposure(self) -> float:
        return sum(p.market_value for p in self.positions.values() if p.is_open)

    def leverage(self) -> float:
        eq = self.equity()
        return self.gross_exposure() / eq if eq > 0 else float("inf")

    @property
    def buying_power(self) -> float:
        """Cash-account semantics for v0: uncommitted cash. No margin."""
        return max(0.0, self.cash)

    def exposure_by_underlying(self) -> dict[str, float]:
        out: dict[str, float] = {}
        short_groups = {
            option_group_key(p.instrument)
            for p in self.positions.values()
            if p.is_open and isinstance(p.instrument, Option) and p.qty < 0
        }
        for p in self.positions.values():
            if not p.is_open:
                continue
            u = p.instrument.underlying
            if isinstance(p.instrument, Option):
                expo = option_exposure(
                    qty=p.qty, multiplier=p.multiplier,
                    delta=(p.greeks.delta if p.greeks is not None else None),
                    underlying_price=p.underlying_mark,
                    premium=(p.mark if p.mark is not None else p.avg_cost),
                    fallback=p.market_value,
                    group_has_short=option_group_key(p.instrument) in short_groups,
                )
            else:
                expo = p.market_value
            out[u] = out.get(u, 0.0) + expo
        return out

    def exposure_by_sector(self, sector_of: Callable[[str], str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for underlying, expo in self.exposure_by_underlying().items():
            sector = sector_of(underlying)
            out[sector] = out.get(sector, 0.0) + expo
        return out

    def net_greeks(self) -> PortfolioGreeks:
        dollar_delta = 0.0
        gamma_shares = 0.0
        theta = 0.0
        vega = 0.0
        for p in self.positions.values():
            if not p.is_open:
                continue
            if isinstance(p.instrument, Equity):
                price = p.mark if p.mark is not None else p.avg_cost
                dollar_delta += p.qty * price
            elif isinstance(p.instrument, Option):
                if p.greeks is None:
                    continue
                mult = p.qty * p.multiplier
                under = p.underlying_mark or 0.0
                dollar_delta += p.greeks.delta * mult * under
                gamma_shares += p.greeks.gamma * mult
                theta += option_theta_per_day(
                    qty=p.qty, multiplier=p.multiplier, theta=p.greeks.theta,
                    premium=(p.mark if p.mark is not None else p.avg_cost))
                vega += p.greeks.vega * mult
        return PortfolioGreeks(
            dollar_delta=dollar_delta, gamma_shares=gamma_shares,
            theta_per_day=theta, vega=vega,
        )

    # ── P&L ──────────────────────────────────────────────────────────────

    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values() if p.is_open)

    def total_pnl(self) -> float:
        """Net of ALL costs: simply equity vs starting equity."""
        return self.equity() - self.starting_equity

    def start_new_day(self) -> None:
        self.day_anchor_equity = self.equity()

    def day_pnl(self) -> float:
        if self.day_anchor_equity is None:
            return 0.0
        return self.equity() - self.day_anchor_equity

    def drawdown(self) -> float:
        """Current drawdown fraction from peak equity (0 = at peak)."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity() / self.peak_equity)

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.is_open]
