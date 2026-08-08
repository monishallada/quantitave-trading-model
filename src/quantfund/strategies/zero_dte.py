"""0DTE sleeve — same-session expiry options on liquid index ETFs.

⚠️  UNVALIDATED BY CONSTRUCTION. Read this before funding it.

The platform's backtest engine is daily with t+1 execution. A 0DTE option is
born and dies inside one session, so **this strategy cannot be walk-forward
validated by the existing harness** — unlike every other sleeve here, its
numbers have never been tested out-of-sample. Validating it needs historical
INTRADAY option quotes (ThetaData/ORATS class data), which the free tier does
not provide.

What the platform's own evidence says about the neighbouring idea: the
`convexity` sleeve (short-dated LONG options) failed six consecutive
out-of-sample validations, hit rate 5-17%. 0DTE is that same trade with the
time value maximised and the recovery time removed — 100% of a 0DTE premium is
time value, and it decays to zero within hours. Treat this sleeve as an
explicitly-capped experiment, not as an edge.

Design given real constraints:
- Alpaca's indicative feed returns NO greeks/IV on 0DTE contracts, so contracts
  are chosen by MONEYNESS (a small % OTM), not delta.
- Signals are intraday only (1-min bars): trend vs session VWAP, short-horizon
  momentum, and a volume-surge confirmation. Daily-bar signals are meaningless
  for a position whose horizon is hours.
- Hard bracket on every position: take-profit, stop-loss, and a FORCED FLAT
  before the close — a 0DTE contract held to expiry is a coin flip between
  intrinsic value and total loss, so the sleeve never holds one into the bell.
- No entries in the final stretch of the session (terminal gamma), and a daily
  trade budget so a chopping tape cannot bleed the sleeve through commissions.

The friction hurdle, measured on the first live fill (21 QQQ 0DTE calls):
round-trip option fees are ~1.4% of premium, and the sleeve pays the full
spread (buys at the ask, marks/sells at the bid). With a +60%/-35% bracket
that puts BREAKEVEN WIN RATE at 40-49% depending on the contract's spread
(36.8% would be breakeven with zero friction). For reference, `convexity` --
the only comparable sleeve that has actually been measured here -- ran a
5-17% hit rate. Nothing about that gap is reassuring; it is the honest bar
this sleeve has to clear.
"""
from __future__ import annotations

import logging
from datetime import time as dtime, timezone

from quantfund.core.instruments import Option, OptionRight
from quantfund.core.snapshot import MarketSnapshot
from quantfund.strategies.base import Signal, SignalAction, SleeveContext, Strategy

log = logging.getLogger(__name__)

# US equity options trade 09:30-16:00 ET == 13:30-20:00 UTC (EDT).
_SESSION_OPEN_UTC = dtime(13, 30)
_SESSION_CLOSE_UTC = dtime(20, 0)


class ZeroDTEStrategy(Strategy):
    """Same-session (0DTE) options sleeve. UNVALIDATED: the daily t+1 backtest
    engine cannot test a position whose entire life is inside one session, so
    this sleeve ships with no out-of-sample evidence. Hard-capped by design."""

    strategy_id = "zero_dte"
    name = "0DTE Intraday (UNVALIDATED)"
    warmup_bars = 0          # intraday sleeve; daily bars are not used
    uses_options = True
    fast_cadence = True      # runner evaluates this sleeve every loop

    def __init__(self,
                 underlyings: list[str] | None = None,
                 otm_pct: float = 0.002,          # ~0.2% OTM strike selection
                 momentum_minutes: int = 15,
                 momentum_threshold: float = 0.0012,
                 volume_surge: float = 1.3,
                 baseline_minutes: int = 30,
                 take_profit: float = 0.60,       # +60% of premium
                 stop_loss: float = 0.35,         # -35% of premium
                 premium_per_trade_pct: float = 0.06,
                 max_concurrent: int = 3,
                 max_entries_per_day: int = 12,
                 no_entry_after_utc: dtime = dtime(18, 45),   # ~14:45 ET
                 force_flat_after_utc: dtime = dtime(19, 30)):  # ~15:30 ET
        self.underlyings = underlyings or ["SPY", "QQQ"]
        self.otm_pct = otm_pct
        self.momentum_minutes = momentum_minutes
        self.momentum_threshold = momentum_threshold
        self.volume_surge = volume_surge
        self.baseline_minutes = baseline_minutes
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.premium_per_trade_pct = premium_per_trade_pct
        self.max_concurrent = max_concurrent
        self.max_entries_per_day = max_entries_per_day
        self.no_entry_after_utc = no_entry_after_utc
        self.force_flat_after_utc = force_flat_after_utc
        self._entries_today = 0
        self._entry_day = None

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_zero_dte(inst: Option, snapshot: MarketSnapshot) -> bool:
        return inst.expiration == snapshot.as_of.astimezone(timezone.utc).date()

    def _session_time(self, snapshot: MarketSnapshot) -> dtime:
        return snapshot.as_of.astimezone(timezone.utc).time()

    def _intraday_signal(self, symbol: str,
                         snapshot: MarketSnapshot) -> tuple[int, dict]:
        """Returns (+1 bullish / -1 bearish / 0 flat, detail)."""
        bars = snapshot.get_intraday_bars(symbol)
        if len(bars) < max(self.momentum_minutes, self.baseline_minutes) + 6:
            return 0, {"reason": "insufficient intraday history",
                       "bars": len(bars)}
        closes = [b.close for b in bars]
        last = closes[-1]
        ref = closes[-(self.momentum_minutes + 1)]
        mom = (last / ref - 1.0) if ref > 0 else 0.0
        vwap = snapshot.session_vwap(symbol)
        # Relative volume against a TRAILING MEDIAN baseline that excludes the
        # recent bars themselves. Two things this must not do:
        #   * include the opening burst in the baseline — intraday volume is
        #     U-shaped, so a session-wide mean makes the ratio < 1.0 for most
        #     of the day and the filter never fires (observed live 2026-08-07:
        #     surge read 0.34-0.81 across SPY/QQQ/IWM at 10:39 ET);
        #   * use a mean — one opening bar can be 20x a midday bar and would
        #     dominate the average. The median is robust to that.
        recent_vol = sum(b.volume for b in bars[-5:]) / 5.0
        baseline_bars = bars[-(self.baseline_minutes + 5):-5]
        base_vols = sorted(b.volume for b in baseline_bars)
        base_vol = (base_vols[len(base_vols) // 2] if base_vols else 0.0)
        surge = (recent_vol / base_vol) if base_vol > 0 else 0.0
        detail = {"last": last, "vwap": vwap, f"mom_{self.momentum_minutes}m": mom,
                  "volume_surge": round(surge, 2), "bars": len(bars)}
        if vwap is None or surge < self.volume_surge:
            detail["reason"] = "no volume confirmation"
            return 0, detail
        if mom > self.momentum_threshold and last > vwap:
            detail["reason"] = "momentum up + above VWAP + volume surge"
            return 1, detail
        if mom < -self.momentum_threshold and last < vwap:
            detail["reason"] = "momentum down + below VWAP + volume surge"
            return -1, detail
        detail["reason"] = "no directional edge"
        return 0, detail

    # ── exits (always evaluated first) ───────────────────────────────────

    def _manage(self, snapshot: MarketSnapshot,
                ctx: SleeveContext) -> tuple[list[Signal], set[str], int]:
        signals: list[Signal] = []
        held_underlyings: set[str] = set()
        open_count = 0
        now = self._session_time(snapshot)
        for pos in ctx.positions.values():
            inst = pos.instrument
            if not (isinstance(inst, Option) and pos.is_open and pos.qty > 0):
                continue
            open_count += 1
            held_underlyings.add(inst.underlying_symbol)
            entry = pos.avg_cost
            mark = pos.mark if pos.mark is not None else entry
            reason = None
            if entry > 0 and mark >= entry * (1.0 + self.take_profit):
                reason = (f"take profit: {mark:.2f} >= entry {entry:.2f} "
                          f"+{self.take_profit:.0%}")
            elif entry > 0 and mark <= entry * (1.0 - self.stop_loss):
                reason = (f"stop loss: {mark:.2f} <= entry {entry:.2f} "
                          f"-{self.stop_loss:.0%}")
            elif now >= self.force_flat_after_utc:
                reason = ("forced flat before the close — a 0DTE contract held "
                          "to expiry is total loss if it finishes OTM")
            if reason:
                open_count -= 1
                # Do NOT release the underlying here. Emitting a CLOSE is not
                # the same as being flat: the order can be rejected, fail, or
                # simply not fill this pass. Releasing on intent let the sleeve
                # open an opposing contract on the same underlying in the SAME
                # pass — on 2026-08-07 it bought a QQQ put while its QQQ call
                # was still open (and, as it happened, while that call's
                # stop-loss was being rejected), leaving an unintended
                # straddle. The symbol frees up on a later pass once the
                # position is actually gone.
                signals.append(Signal(
                    instrument=inst, action=SignalAction.CLOSE, target_weight=0.0,
                    rationale=f"0DTE {reason}", strategy_id=self.strategy_id,
                    ts=snapshot.as_of, confidence=0.9,
                    rationale_payload={"reason": reason, "entry": entry,
                                       "mark": mark},
                ))
        return signals, held_underlyings, open_count

    # ── Strategy interface ───────────────────────────────────────────────

    def generate_signals(self, snapshot: MarketSnapshot,
                         ctx: SleeveContext) -> list[Signal]:
        signals, held, open_count = self._manage(snapshot, ctx)

        day = snapshot.as_of.astimezone(timezone.utc).date()
        if self._entry_day != day:
            self._entry_day = day
            self._entries_today = 0

        now = self._session_time(snapshot)
        if now < _SESSION_OPEN_UTC or now >= self.no_entry_after_utc:
            return signals          # outside the entry window
        if open_count >= self.max_concurrent:
            return signals
        if self._entries_today >= self.max_entries_per_day:
            return signals
        if ctx.allocated_capital <= 0:
            return signals

        for sym in self.underlyings:
            if open_count >= self.max_concurrent:
                break
            if sym in held:
                continue
            chain = snapshot.get_option_chain(sym)
            if chain is None or chain.underlying_price <= 0:
                continue
            direction, detail = self._intraday_signal(sym, snapshot)
            if direction == 0:
                continue
            spot = chain.underlying_price
            right = OptionRight.CALL if direction > 0 else OptionRight.PUT
            target_strike = (spot * (1.0 + self.otm_pct) if direction > 0
                             else spot * (1.0 - self.otm_pct))
            # 0DTE contracts only, chosen by MONEYNESS (no greeks on this feed)
            todays = [q for q in chain.contracts(right=right)
                      if self._is_zero_dte(q.instrument, snapshot)
                      and q.quote.bid > 0 and q.quote.ask > 0]
            if not todays:
                continue
            oq = min(todays, key=lambda q: abs(q.instrument.strike - target_strike))
            mid = oq.quote.mid
            if mid <= 0:
                continue
            # reject contracts whose spread eats the edge
            if oq.quote.spread / mid > 0.10:
                continue
            budget = self.premium_per_trade_pct * ctx.allocated_capital
            contracts = int(budget // (mid * 100.0))
            if contracts < 1:
                continue
            weight = contracts * mid * 100.0 * 1.001 / ctx.allocated_capital
            if weight > 1.0:
                continue
            self._entries_today += 1
            open_count += 1
            signals.append(Signal(
                instrument=oq.instrument,
                action=SignalAction.OPEN_LONG,
                target_weight=weight,
                rationale=(
                    f"0DTE {'CALL' if direction > 0 else 'PUT'} {sym}: "
                    f"{detail['reason']}; {self.momentum_minutes}m mom "
                    f"{detail[f'mom_{self.momentum_minutes}m']:+.2%}, "
                    f"vol surge {detail['volume_surge']}x, "
                    f"strike {oq.instrument.strike:g} vs spot {spot:.2f}, "
                    f"{contracts}x @ ~{mid:.2f} (max loss = premium; "
                    f"TP +{self.take_profit:.0%} / SL -{self.stop_loss:.0%} / "
                    f"flat by {self.force_flat_after_utc} UTC)"),
                strategy_id=self.strategy_id,
                ts=snapshot.as_of,
                confidence=0.4,   # deliberately low: this sleeve is unvalidated
                rationale_payload={
                    "underlying": sym, "direction": "long" if direction > 0 else "short",
                    "right": right.value, "strike": oq.instrument.strike,
                    "spot": spot, "premium_mid": mid,
                    "spread_pct": round(oq.quote.spread / mid, 4),
                    "contracts": contracts,
                    "max_loss": contracts * mid * 100.0,
                    "signal": detail,
                    "entries_today": self._entries_today,
                    "exits": {"take_profit": self.take_profit,
                              "stop_loss": self.stop_loss,
                              "force_flat_utc": str(self.force_flat_after_utc)},
                    "VALIDATION": "NONE — daily backtest engine cannot test 0DTE",
                },
            ))
        return signals
