"""Pre-trade risk gate. Every order passes through RiskManager.check_order
before it can reach a broker. Conservative by default: undefined-risk (naked
short) option positions are rejected unless explicitly enabled in RiskLimits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from quantfund.core.config import RiskLimits, sector_of as default_sector_of
from quantfund.core.instruments import AssetClass, Equity, Option, OptionRight
from quantfund.core.orders import Order, OrderSide
from quantfund.core.portfolio import Portfolio
from quantfund.core.snapshot import MarketSnapshot


@dataclass
class RiskDecision:
    approved: bool
    reason: str = ""
    adjusted_qty: Optional[float] = None  # set when the order was downsized


def _leg_price(leg: OrderLeg, snapshot: MarketSnapshot) -> Optional[float]:
    inst = leg.instrument
    if isinstance(inst, Equity):
        return snapshot.last_price(inst.symbol)
    if isinstance(inst, Option):
        chain = snapshot.get_option_chain(inst.underlying_symbol)
        if chain is None:
            return None
        oq = next((q for q in chain.quotes if q.instrument.symbol == inst.symbol), None)
        if oq is None or oq.quote.mid <= 0:
            return None
        return oq.quote.mid
    return None


def _leg_greeks(leg: OrderLeg, snapshot: MarketSnapshot):
    inst = leg.instrument
    if not isinstance(inst, Option):
        return None
    chain = snapshot.get_option_chain(inst.underlying_symbol)
    if chain is None:
        return None
    oq = next((q for q in chain.quotes if q.instrument.symbol == inst.symbol), None)
    return oq.greeks if oq else None


class RiskManager:
    def __init__(self, limits: RiskLimits,
                 sector_of: Callable[[str], str] = default_sector_of):
        self.limits = limits
        self.sector_of = sector_of

    # ── naked-short detection ────────────────────────────────────────────

    def _undefined_risk_reason(self, order: Order,
                               portfolio: Portfolio) -> Optional[str]:
        """Post-trade net-coverage check across the WHOLE order.

        Projects every option contract's post-trade quantity, groups by
        (underlying, right, expiration), and nets longs against shorts within
        each group — so one long contract can cover exactly one short contract,
        and a BUY that merely closes an existing short contributes zero cover.
        Residual uncovered shorts must be covered by shares (calls) or cash
        (puts); both are CONSUMED across groups, never double-counted.
        Returns a rejection reason, or None if no undefined risk remains.
        """
        def uncovered_by_group(with_order: bool) -> dict[tuple, float]:
            book: dict[str, tuple[Option, float]] = {}
            for key, pos in portfolio.positions.items():
                if isinstance(pos.instrument, Option) and pos.is_open:
                    book[key] = (pos.instrument, pos.qty)
            if with_order:
                for leg in order.legs:
                    if not isinstance(leg.instrument, Option):
                        continue
                    inst = leg.instrument
                    _, qty = book.get(inst.key, (inst, 0.0))
                    book[inst.key] = (inst, qty + leg.side.sign * leg.qty)
            groups: dict[tuple, list[tuple[Option, float]]] = {}
            for inst, qty in book.values():
                k = (inst.underlying_symbol, inst.right, inst.expiration)
                groups.setdefault(k, []).append((inst, qty))
            out: dict[tuple, float] = {}
            for k, contracts in groups.items():
                longs = sum(q for _, q in contracts if q > 0)
                shorts = -sum(q for _, q in contracts if q < 0)
                out[k] = max(0.0, shorts - longs)
            return out

        touched_groups = {
            (leg.instrument.underlying_symbol, leg.instrument.right,
             leg.instrument.expiration)
            for leg in order.legs if isinstance(leg.instrument, Option)
        }
        if not touched_groups:
            return None

        before = uncovered_by_group(with_order=False)
        after = uncovered_by_group(with_order=True)

        # Only judge groups THIS ORDER touches, and only reject when the order
        # INCREASES uncovered short exposure. Evaluating the whole book meant a
        # single pre-existing short put that drifted under-collateralized
        # rejected every unrelated option order in the platform — including
        # long calls, which carry no undefined risk at all (2026-08-05).
        shares_available: dict[str, float] = {}
        cash_available = portfolio.cash
        for key in sorted(touched_groups, key=str):
            underlying, right, expiration = key
            new_uncovered = after.get(key, 0.0)
            old_uncovered = before.get(key, 0.0)
            # Allow only orders that leave the group covered, or that strictly
            # REDUCE its uncovered shorts. A long call touches a group with no
            # shorts (uncovered 0) and always passes; closing a bad short
            # passes because it reduces; rolling into an equally-naked short
            # does not.
            if new_uncovered <= 1e-9 or new_uncovered < old_uncovered - 1e-9:
                continue
            if right == OptionRight.CALL:
                if underlying not in shares_available:
                    pos = portfolio.positions.get(underlying)
                    shares_available[underlying] = pos.qty if pos else 0.0
                need = 100.0 * new_uncovered
                if shares_available[underlying] + 1e-9 >= need:
                    shares_available[underlying] -= need  # consumed
                    continue
                return (f"naked short call {underlying} {expiration}: need "
                        f"{need:.0f} shares, have "
                        f"{shares_available[underlying]:.0f} uncommitted")
            else:
                contracts = [
                    (pos.instrument, pos.qty)
                    for pos in portfolio.positions.values()
                    if isinstance(pos.instrument, Option)
                    and (pos.instrument.underlying_symbol, pos.instrument.right,
                         pos.instrument.expiration) == key
                ] + [
                    (leg.instrument, -leg.qty) for leg in order.legs
                    if isinstance(leg.instrument, Option)
                    and leg.side == OrderSide.SELL
                    and (leg.instrument.underlying_symbol, leg.instrument.right,
                         leg.instrument.expiration) == key
                ]
                short_strikes = [i.strike for i, q in contracts if q < 0]
                if not short_strikes:
                    continue
                reserve = max(short_strikes) * 100.0 * new_uncovered
                if cash_available + 1e-9 >= reserve:
                    cash_available -= reserve  # consumed
                    continue
                return (f"short put {underlying} {expiration} not cash-secured: "
                        f"need ${reserve:,.0f}, have ${cash_available:,.0f} "
                        f"uncommitted")
        return None

    # ── main gate ────────────────────────────────────────────────────────

    def check_order(self, order: Order, portfolio: Portfolio,
                    snapshot: MarketSnapshot) -> RiskDecision:
        lim = self.limits
        equity = portfolio.equity()
        if equity <= 0:
            return RiskDecision(False, "portfolio equity is non-positive")

        # price every leg first
        leg_prices: list[float] = []
        for leg in order.legs:
            px = _leg_price(leg, snapshot)
            if px is None or px <= 0:
                return RiskDecision(False, f"no market data for {leg.instrument.key}")
            leg_prices.append(px)

        # 1) undefined-risk short options
        if not lim.allow_naked_short_options:
            reason = self._undefined_risk_reason(order, portfolio)
            if reason:
                return RiskDecision(False, f"undefined-risk rejected: {reason}")
        # short options without greeks are unpriceable risk — reject regardless
        for leg in order.legs:
            if isinstance(leg.instrument, Option) and leg.side == OrderSide.SELL:
                pos = portfolio.positions.get(leg.instrument.key)
                if (pos is None or pos.qty < leg.qty) and _leg_greeks(leg, snapshot) is None:
                    return RiskDecision(
                        False, f"short option {leg.instrument.key} has no greeks on the "
                               f"chain; refusing unpriceable risk")

        # 2) short equity ban
        if not lim.allow_short_equity:
            for leg in order.legs:
                if isinstance(leg.instrument, Equity) and leg.side == OrderSide.SELL:
                    pos = portfolio.positions.get(leg.instrument.key)
                    held = pos.qty if pos else 0.0
                    if leg.qty > held + 1e-9:
                        return RiskDecision(
                            False, f"short equity disabled: selling {leg.qty:.0f} "
                                   f"{leg.instrument.key} but hold {held:.0f}")

        # 3) per-order notional (downsizable, single-leg only)
        adjusted_qty: Optional[float] = None
        for i, leg in enumerate(order.legs):
            notional = leg_prices[i] * leg.qty * leg.instrument.multiplier
            if notional > lim.max_order_notional:
                if len(order.legs) > 1:
                    return RiskDecision(
                        False, f"order leg notional ${notional:,.0f} > per-order limit "
                               f"${lim.max_order_notional:,.0f} (multi-leg, not downsizable)")
                unit = leg_prices[i] * leg.instrument.multiplier
                new_qty = float(int(lim.max_order_notional / unit))
                if new_qty < 1:
                    return RiskDecision(
                        False, f"order notional ${notional:,.0f} > limit "
                               f"${lim.max_order_notional:,.0f} and cannot downsize below 1")
                adjusted_qty = new_qty

        eff_qty = [adjusted_qty if adjusted_qty is not None else leg.qty
                   for leg in order.legs]

        # 4) per-instrument notional post-trade
        for i, leg in enumerate(order.legs):
            pos = portfolio.positions.get(leg.instrument.key)
            cur_qty = pos.qty if pos else 0.0
            post_qty = cur_qty + leg.side.sign * eff_qty[i]
            post_notional = abs(post_qty) * leg_prices[i] * leg.instrument.multiplier
            if post_notional > lim.max_position_notional:
                return RiskDecision(
                    False, f"position notional ${post_notional:,.0f} for "
                           f"{leg.instrument.key} > limit ${lim.max_position_notional:,.0f}")

        # exposure deltas for caps 5-8.
        # Cap policy: an order is rejected only when it WORSENS a breach —
        # risk-reducing trades must always be allowed, otherwise an over-limit
        # book could never trade back into compliance.
        def worsens(pre: float, post: float, cap: float) -> bool:
            return abs(post) > cap and abs(post) > abs(pre) + 1e-9

        expo_by_under = portfolio.exposure_by_underlying()
        delta_expo: dict[str, float] = {}
        gross_delta = 0.0
        dollar_delta_change = 0.0
        theta_change = 0.0
        vega_change = 0.0
        for i, leg in enumerate(order.legs):
            inst = leg.instrument
            signed_qty = leg.side.sign * eff_qty[i]
            if isinstance(inst, Equity):
                d_expo = signed_qty * leg_prices[i]
                dollar_delta_change += d_expo
            else:
                g = _leg_greeks(leg, snapshot)
                under_px = snapshot.last_price(inst.underlying_symbol) or 0.0
                if g is not None and under_px > 0:
                    d_expo = g.delta * signed_qty * inst.multiplier * under_px
                    dollar_delta_change += d_expo
                    theta_change += g.theta * signed_qty * inst.multiplier
                    vega_change += g.vega * signed_qty * inst.multiplier
                else:
                    # long option without greeks: bound exposure by premium
                    d_expo = signed_qty * leg_prices[i] * inst.multiplier
                    dollar_delta_change += d_expo
            u = inst.underlying
            delta_expo[u] = delta_expo.get(u, 0.0) + d_expo
            # gross change = |post-trade qty| - |current qty|, priced
            pos = portfolio.positions.get(inst.key)
            cur_qty = pos.qty if pos else 0.0
            gross_delta += (abs(cur_qty + signed_qty) - abs(cur_qty)) \
                * leg_prices[i] * inst.multiplier

        # 5) per-underlying cap
        for u, d in delta_expo.items():
            pre = expo_by_under.get(u, 0.0)
            post = pre + d
            cap = lim.max_position_pct * equity
            if worsens(pre, post, cap):
                return RiskDecision(
                    False, f"underlying exposure ${abs(post):,.0f} for {u} > "
                           f"{lim.max_position_pct:.0%} of equity (${cap:,.0f})")

        # 6) per-sector cap
        sector_expo = portfolio.exposure_by_sector(self.sector_of)
        sector_delta: dict[str, float] = {}
        for u, d in delta_expo.items():
            s = self.sector_of(u)
            sector_delta[s] = sector_delta.get(s, 0.0) + d
        for s, d in sector_delta.items():
            pre = sector_expo.get(s, 0.0)
            post = pre + d
            cap = lim.max_sector_pct * equity
            if worsens(pre, post, cap):
                return RiskDecision(
                    False, f"sector exposure ${abs(post):,.0f} for {s} > "
                           f"{lim.max_sector_pct:.0%} of equity (${cap:,.0f})")

        # 7) gross leverage
        pre_gross = portfolio.gross_exposure()
        post_gross = pre_gross + gross_delta
        if worsens(pre_gross, post_gross, lim.max_gross_leverage * equity):
            return RiskDecision(
                False, f"gross exposure ${post_gross:,.0f} > "
                       f"{lim.max_gross_leverage:.1f}x equity")

        # 8) net dollar delta / theta / vega
        g0 = portfolio.net_greeks()
        post_delta = g0.dollar_delta + dollar_delta_change
        if worsens(g0.dollar_delta, post_delta, lim.max_net_delta_pct * equity):
            return RiskDecision(
                False, f"net dollar delta ${post_delta:,.0f} > "
                       f"{lim.max_net_delta_pct:.0%} of equity")
        post_theta = g0.theta_per_day + theta_change
        if worsens(g0.theta_per_day, post_theta,
                   lim.max_theta_pct_per_day * equity):
            return RiskDecision(
                False, f"portfolio theta ${post_theta:,.2f}/day > "
                       f"{lim.max_theta_pct_per_day:.2%} of equity")
        post_vega = g0.vega + vega_change
        if worsens(g0.vega, post_vega, lim.max_vega_pct * equity):
            return RiskDecision(
                False, f"portfolio vega ${post_vega:,.2f} > "
                       f"{lim.max_vega_pct:.2%} of equity")

        # 9) TOTAL long-option premium at risk (the portfolio-level version of
        #    "loss is capped at premium": per-position caps do not stop every
        #    contract expiring worthless at once)
        premium_at_risk = 0.0
        for pos in portfolio.open_positions():
            if isinstance(pos.instrument, Option) and pos.qty > 0:
                premium_at_risk += pos.notional
        added_premium = 0.0
        for i, leg in enumerate(order.legs):
            if isinstance(leg.instrument, Option) and leg.side == OrderSide.BUY:
                pos = portfolio.positions.get(leg.instrument.key)
                cur = pos.qty if pos else 0.0
                opening = max(0.0, min(eff_qty[i], cur + eff_qty[i]))
                added_premium += (opening * leg_prices[i]
                                  * leg.instrument.multiplier)
        cap_premium = lim.max_long_option_premium_pct * equity
        if added_premium > 0 and premium_at_risk + added_premium > cap_premium:
            return RiskDecision(
                False,
                f"long-option premium at risk "
                f"${premium_at_risk + added_premium:,.0f} > "
                f"{lim.max_long_option_premium_pct:.0%} of equity "
                f"(${cap_premium:,.0f}) — total premium that can expire "
                f"worthless is bounded")

        # 10) cash buffer on net-debit orders — measured against UNCOMMITTED
        #     cash. Cash securing existing short puts is collateral, not
        #     spendable: letting an equity buy consume it silently
        #     under-collateralizes the put book (2026-08-05: stock purchases
        #     drained cash below the NVDA put reserve and froze options
        #     trading platform-wide).
        committed_cash = 0.0
        for pos in portfolio.open_positions():
            inst = pos.instrument
            if (isinstance(inst, Option) and inst.right == OptionRight.PUT
                    and pos.qty < 0):
                committed_cash += inst.strike * 100.0 * abs(pos.qty)
        cash_out = 0.0
        for i, leg in enumerate(order.legs):
            cash_out += leg.side.sign * eff_qty[i] * leg_prices[i] * leg.instrument.multiplier
        if cash_out > 0:
            post_cash = portfolio.cash - committed_cash - cash_out
            if post_cash < lim.min_cash_buffer_pct * equity:
                return RiskDecision(
                    False, f"cash buffer: post-trade uncommitted cash "
                           f"${post_cash:,.0f} < {lim.min_cash_buffer_pct:.0%} "
                           f"of equity (${committed_cash:,.0f} is securing "
                           f"short puts)")

        if adjusted_qty is not None:
            return RiskDecision(True, "downsized to per-order notional limit",
                                adjusted_qty=adjusted_qty)
        return RiskDecision(True, "ok")
