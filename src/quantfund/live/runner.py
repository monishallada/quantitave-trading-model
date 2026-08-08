"""The live paper-trading loop: snapshot → sync → breakers → strategies →
risk gate → orders → fills → state. One iteration never crashes the loop."""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from quantfund.allocation.allocator import CapitalAllocator, SleevePerf
from quantfund.core.config import Settings, sector_of
from quantfund.core.instruments import Equity, Option
from quantfund.core.orders import Order, OrderSide, OrderStatus, single_leg
from quantfund.core.portfolio import Portfolio
from quantfund.core.snapshot import MarketSnapshot
from quantfund.core.state import PlatformState, StrategyStatus, TradeRecord, utcnow
from quantfund.data.provider import DataProvider
from quantfund.execution.broker import Broker, BrokerError
from quantfund.risk.circuit_breakers import CircuitBreakerBoard
from quantfund.risk.kill_switch import KillSwitch
from quantfund.risk.limits import RiskManager
from quantfund.strategies.base import Signal, SignalAction, SleeveContext, Strategy

log = logging.getLogger(__name__)

_TRANSIENT_MARKERS = (
    "connectionerror", "maxretryerror", "nameresolutionerror", "timeout",
    "remotedisconnected", "remote end closed", "connection aborted",
    "connection reset", "connection refused", "network is unreachable",
    "temporary failure in name resolution", "ssl", "read timed out",
    "502", "503", "504",
)


def _is_transient(exc: BaseException) -> bool:
    """Network/connectivity blip rather than a logic fault. These must not
    halt trading at the same rate as real errors (2026-08-04: a wifi drop
    produced 13 consecutive ConnectionErrors and tripped error_burst)."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(m in text for m in _TRANSIENT_MARKERS)


class LiveRunner:
    def __init__(self, settings: Settings, broker: Broker, provider: DataProvider,
                 strategies: list[Strategy], state: PlatformState,
                 risk_manager: Optional[RiskManager] = None,
                 breakers: Optional[CircuitBreakerBoard] = None,
                 allocator: Optional[CapitalAllocator] = None,
                 kill_switch: Optional[KillSwitch] = None):
        self.settings = settings
        self.broker = broker
        self.provider = provider
        self.strategies = strategies
        self.state = state
        self.risk = risk_manager or RiskManager(settings.risk)
        self.breakers = breakers or CircuitBreakerBoard(settings.risk, state)
        self.allocator = allocator or CapitalAllocator(settings.risk)
        self.kill = kill_switch or KillSwitch(state, broker)

        self.portfolio = Portfolio(cash=settings.starting_cash)  # aggregate mirror
        self.sleeves: dict[str, Portfolio] = {
            s.strategy_id: Portfolio(cash=0.0, starting_equity=1.0)
            for s in strategies
        }
        self.sleeve_capital: dict[str, float] = {}
        self.sleeve_returns: dict[str, list[float]] = {
            s.strategy_id: [] for s in strategies
        }
        self._prev_sleeve_eq: dict[str, float] = {}
        self._order_meta: dict[str, tuple[str, str]] = {}  # broker id -> (sid, rationale)
        self._seen_fills: set[tuple] = set()
        # fill-derived local book for reconciliation: seeded from the broker on
        # first sync, then evolved ONLY by ingested fills — so a divergence
        # from broker truth (missed fills, manual trades) actually trips the
        # reconciliation breaker instead of comparing the broker to itself.
        self._local_qty: dict[str, float] = {}
        self._local_seeded = False
        # contracts the expiry guard has already sent a flatten for today —
        # re-sending while the first order works would double-sell into a short
        self._expiry_guard_sent: set[str] = set()
        self._expiry_guard_day = None
        self._returns_day = None
        self._last_rebalance: Optional[datetime] = None
        self._current_day = None
        self._flattened_by_breaker = False
        self._killed_handled = False

    # ── broker sync ──────────────────────────────────────────────────────

    def _sync_from_broker(self, snapshot: MarketSnapshot) -> None:
        account = self.broker.get_account()
        broker_positions = self.broker.get_positions()
        prev = self.portfolio
        fresh = Portfolio(cash=account.cash,
                          starting_equity=prev.starting_equity or account.equity)
        fresh.peak_equity = max(prev.peak_equity, account.equity)
        fresh.day_anchor_equity = prev.day_anchor_equity
        fresh.total_costs = prev.total_costs
        fresh.total_commissions_fees = prev.total_commissions_fees
        for bp in broker_positions:
            from quantfund.core.portfolio import Position
            fresh.positions[bp.instrument.key] = Position(
                instrument=bp.instrument, qty=bp.qty,
                avg_cost=bp.avg_entry_price,
                mark=(bp.market_value / (bp.qty * bp.instrument.multiplier))
                if bp.qty else bp.avg_entry_price,
                mark_ts=snapshot.as_of,
            )
        fresh.mark_from_snapshot(snapshot, risk_free=self.settings.risk_free_rate)
        # Cumulative costs come from the durable trade log EVERY sync, not from
        # this process's ingested fills. _ingest_fills applies fills to the
        # SLEEVES; the master portfolio is rebuilt from broker positions each
        # loop, so its total_costs was never incremented at all — the
        # dashboard's "costs paid" tile has always read $0.00 while real fees
        # were being paid ($62.01 on 2026-08-07).
        try:
            cf, tot = self.state.total_costs_recorded()
            fresh.total_commissions_fees = cf
            fresh.total_costs = tot
        except Exception as e:  # noqa: BLE001
            self.breakers.record_error("seed_costs", repr(e))
        if not self._local_seeded:
            # first sync of this process: trust the broker as the baseline for
            # BOTH the reconciliation book and the P&L/drawdown anchors —
            # config starting_cash is not the account's real equity.
            self._local_qty = {bp.instrument.key: bp.qty for bp in broker_positions}
            self._local_seeded = True
            # Pre-existing fills are ALREADY reflected in those positions —
            # mark them seen so _ingest_fills doesn't apply them a second time
            # (double-counting used to trip the reconciliation breaker on every
            # restart into a non-empty account).
            try:
                for f in self.broker.get_fills(since=None):
                    self._seen_fills.add((f.order_id, f.instrument.key,
                                          f.side.value, f.qty, f.ts.isoformat()))
            except BrokerError as e:
                self.breakers.record_error("broker.get_fills(seed)", str(e))
            fresh.starting_equity = account.equity
            fresh.peak_equity = account.equity
            self._reattribute_sleeves(broker_positions, snapshot)
        self.portfolio = fresh
        self._broker_positions = broker_positions
        # Re-mark EVERY sleeve every loop. Sleeve positions are only touched by
        # apply_fill, which stamps mark = fill price — so without this a
        # sleeve's mark is frozen at its entry forever and every mark-based
        # exit is dead code: 0DTE take-profit/stop-loss, put_income's
        # profit-take and loss-stop. The QQQ 0DTE call reached +60.3% on the
        # master book on 2026-08-07 while its sleeve still saw the 0.92 entry,
        # so the take-profit never fired. Sleeve equity feeds the allocator's
        # Sharpe scoring too, so stale marks quietly corrupt allocation.
        for sleeve in self.sleeves.values():
            sleeve.mark_from_snapshot(snapshot,
                                      risk_free=self.settings.risk_free_rate)

    def _reattribute_sleeves(self, broker_positions, snapshot) -> None:
        """Rebuild sleeve ownership of pre-existing positions after a restart.

        Ownership normally lives in ``_order_meta``, which is in-memory and
        dies with the process. The master book is rebuilt from the broker, but
        the sleeves came back EMPTY, so every open position was orphaned:

          * nobody ran put_income's profit-take / loss-stop / time exits,
          * nobody rolled momentum's ITM calls,
          * a 0DTE contract open across a restart lost its TP/SL brackets
            (the expiry guard still caught it, but only at the close),
          * and worst, each sleeve saw ``current = 0`` for names it already
            held, so it re-requested the full target and doubled up until the
            risk gate rejected it — the exposure-rejection spam in the event
            log was this bug, not a limit that was too tight.

        The trades table already persists strategy_id per instrument, so
        ownership is recoverable. Positions are seeded from BROKER truth (not
        replayed history) so a sleeve's book can never contradict the account.
        """
        try:
            owners: dict[str, str] = {}
            for t in self.state.get_trades(limit=5000):   # newest first
                owners.setdefault(t["instrument_key"], t["strategy_id"])
        except Exception as e:  # noqa: BLE001
            self.breakers.record_error("reattribute", repr(e))
            return
        from quantfund.core.portfolio import Position
        claimed = 0
        for bp in broker_positions:
            sid = owners.get(bp.instrument.key)
            sleeve = self.sleeves.get(sid) if sid else None
            if sleeve is None:
                self.state.log_event(
                    "warning", "reattribute",
                    f"{bp.instrument.key} ({bp.qty:+g}) has no known sleeve — "
                    "no strategy will manage its exit")
                continue
            sleeve.positions[bp.instrument.key] = Position(
                instrument=bp.instrument, qty=bp.qty,
                avg_cost=bp.avg_entry_price,
                mark=(bp.market_value / (bp.qty * bp.instrument.multiplier))
                if bp.qty else bp.avg_entry_price,
                mark_ts=snapshot.as_of,
            )
            claimed += 1
        for sleeve in self.sleeves.values():
            sleeve.mark_from_snapshot(snapshot,
                                      risk_free=self.settings.risk_free_rate)
        if claimed:
            self.state.log_event(
                "info", "reattribute",
                f"restored {claimed}/{len(broker_positions)} positions to "
                "their owning sleeves")

    # ── signal → order ───────────────────────────────────────────────────

    def _signal_to_order(self, sig: Signal, sleeve: Portfolio,
                         snapshot: MarketSnapshot) -> Optional[Order]:
        inst = sig.instrument
        if sig.action == SignalAction.CLOSE:
            pos = sleeve.positions.get(inst.key)
            if pos is None or not pos.is_open:
                return None
            side = OrderSide.SELL if pos.qty > 0 else OrderSide.BUY
            return single_leg(inst, side, abs(pos.qty), strategy_id=sig.strategy_id)
        capital = self.sleeve_capital.get(sig.strategy_id, 0.0)
        if capital <= 0:
            return None
        if isinstance(inst, Equity):
            price = snapshot.last_price(inst.symbol)
            mult = 1
        elif isinstance(inst, Option):
            chain = snapshot.get_option_chain(inst.underlying_symbol)
            oq = None if chain is None else next(
                (q for q in chain.quotes if q.instrument.symbol == inst.symbol), None)
            price = oq.quote.mid if oq else None
            mult = inst.multiplier
        else:
            return None
        if price is None or price <= 0:
            return None
        target_dollars = sig.target_weight * capital
        pos = sleeve.positions.get(inst.key)
        current = (pos.qty * price * mult) if pos else 0.0
        delta = target_dollars - current
        # No-trade band: ignore drift under 12% of target. Widened to 25% once
        # to stop churn (2026-08-03: 85 equity trades in one session); halved
        # again deliberately for a faster, more actively-rebalanced book. This
        # buys turnover with friction — max_trades_per_hour is the backstop.
        if current != 0.0 and abs(delta) < 0.12 * abs(target_dollars):
            return None
        qty = int(abs(delta) / (price * mult))
        if qty < 1:
            return None
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        return single_leg(inst, side, float(qty), strategy_id=sig.strategy_id)

    # ── expiry guard ─────────────────────────────────────────────────────

    def _close_expiring_options(self, snapshot: MarketSnapshot) -> int:
        """Flatten every long option expiring TODAY once past the cutoff.

        A sleeve's own forced-flat only fires if that sleeve runs: it can be
        starved of capital, throw, be throttled, or have its signals rejected.
        None of that is acceptable for a same-day option, because the failure
        is not symmetric:

          * expires OTM -> worthless. Loss capped at premium. Survivable.
          * expires ITM -> Alpaca AUTO-EXERCISES. Each contract assigns 100
            shares. Twelve SPY contracts is ~$900k of stock delivered into a
            $100k account overnight — a leverage event far larger than any
            risk limit this platform enforces.

        So this runs unconditionally, every loop, before the breakers and
        outside the rebalance throttle, for options owned by ANY sleeve.

        SUBMIT ONCE PER CONTRACT PER DAY. The broker position does not clear
        the instant the order is sent, so a naive "flatten anything still
        showing" would re-send every 60s while the first sell was still
        working — selling 21 contracts twice turns a long call into a NAKED
        SHORT, the one exposure this platform bans outright. Being late to
        re-try is recoverable; accidentally going short 0DTE is not.
        """
        now = utcnow()
        if now.time() < self.settings.expiry_flatten_after_utc:
            return 0
        today = now.date()
        if self._expiry_guard_day != today:
            self._expiry_guard_day = today
            self._expiry_guard_sent.clear()
        closed = 0
        for pos in list(self.portfolio.positions.values()):
            inst = pos.instrument
            if not (isinstance(inst, Option) and pos.is_open):
                continue
            if inst.key in self._expiry_guard_sent:
                continue
            if inst.expiration != today:
                continue
            side = OrderSide.SELL if pos.qty > 0 else OrderSide.BUY
            order = single_leg(inst, side, abs(pos.qty),
                               strategy_id="expiry_guard")
            try:
                self.broker.submit_order(order)
            except BrokerError as e:
                self.breakers.record_error("expiry_guard", str(e))
                self.state.log_event(
                    "critical", "expiry_guard",
                    f"COULD NOT flatten {inst.symbol} expiring today: {e}")
                continue
            # mark sent ONLY after the broker accepted it — a submit that
            # raised must be retried next loop, not silently abandoned
            self._expiry_guard_sent.add(inst.key)
            closed += 1
            self.state.log_event(
                "warning", "expiry_guard",
                f"flattened {inst.symbol} ({pos.qty:+g}) expiring {today} — "
                f"never hold a 0DTE contract into the close")
        return closed

    # ── fills → accounting ───────────────────────────────────────────────

    def _ingest_fills(self) -> int:
        # Poll everything and dedupe locally — timestamp-based "since" filters
        # race with fill timestamps (a sim fill is stamped at snapshot.as_of,
        # which predates the poll clock).
        try:
            fills = self.broker.get_fills(since=None)
        except BrokerError as e:
            self.breakers.record_error("broker.get_fills", str(e))
            return 0
        n = 0
        for f in fills:
            key = (f.order_id, f.instrument.key, f.side.value, f.qty,
                   f.ts.isoformat())
            if key in self._seen_fills:
                continue
            self._seen_fills.add(key)
            new_qty = self._local_qty.get(f.instrument.key, 0.0) \
                + f.side.sign * f.qty
            if abs(new_qty) < 1e-9:
                self._local_qty.pop(f.instrument.key, None)
            else:
                self._local_qty[f.instrument.key] = new_qty
            sid = f.strategy_id
            rationale_id = ""
            if not sid and f.order_id in self._order_meta:
                sid, rationale_id = self._order_meta[f.order_id]
            elif f.order_id in self._order_meta:
                _, rationale_id = self._order_meta[f.order_id]
            sleeve = self.sleeves.get(sid)
            if sleeve is not None:
                sleeve.apply_fill(f)
            self.state.record_trade(TradeRecord(
                ts=f.ts, strategy_id=sid or "unattributed",
                instrument_key=f.instrument.key,
                asset_class=f.instrument.asset_class.value,
                side=f.side.value, qty=f.qty, price=f.price,
                commission=f.commission, fees=f.fees, slippage=f.slippage_cost,
                rationale_id=rationale_id, order_id=f.order_id,
            ))
            self.breakers.record_trade(f.ts)
            n += 1
        return n

    # ── per-loop state publishing ────────────────────────────────────────

    def _publish(self, snapshot: Optional[MarketSnapshot]) -> None:
        g = self.portfolio.net_greeks()
        summary = {
            "equity": self.portfolio.equity(),
            "cash": self.portfolio.cash,
            "buying_power": self.portfolio.buying_power,
            "day_pnl": self.portfolio.day_pnl(),
            "total_pnl": self.portfolio.total_pnl(),
            "realized_pnl": self.portfolio.realized_pnl(),
            "unrealized_pnl": self.portfolio.unrealized_pnl(),
            "gross_exposure": self.portfolio.gross_exposure(),
            "gross_delta_exposure": self.portfolio.gross_delta_exposure(),
            "net_exposure": self.portfolio.net_exposure(),
            "drawdown": self.portfolio.drawdown(),
            "total_costs": self.portfolio.total_costs,
            "greeks": {
                "dollar_delta": g.dollar_delta, "gamma_shares": g.gamma_shares,
                "theta_per_day": g.theta_per_day, "vega": g.vega,
            },
            "as_of": snapshot.as_of.isoformat() if snapshot else None,
        }
        positions = []
        for p in self.portfolio.open_positions():
            row = {
                "key": p.instrument.key,
                "asset_class": p.instrument.asset_class.value,
                "qty": p.qty, "avg_cost": p.avg_cost, "mark": p.mark,
                "notional": p.notional, "unrealized_pnl": p.unrealized_pnl,
            }
            if isinstance(p.instrument, Option):
                row["expiration"] = p.instrument.expiration.isoformat()
                row["strike"] = p.instrument.strike
                row["right"] = p.instrument.right.value
                if p.greeks:
                    row["greeks"] = {
                        "delta": p.greeks.delta, "gamma": p.greeks.gamma,
                        "theta": p.greeks.theta, "vega": p.greeks.vega,
                    }
            positions.append(row)
        self.state.update_portfolio(summary, positions)
        self.state.record_equity(
            summary["equity"], summary["cash"], summary["buying_power"],
            summary["gross_exposure"])
        for s in self.strategies:
            sleeve = self.sleeves[s.strategy_id]
            self.state.update_strategy(StrategyStatus(
                strategy_id=s.strategy_id, name=s.name,
                allocated_capital=self.sleeve_capital.get(s.strategy_id, 0.0),
                weight=(self.sleeve_capital.get(s.strategy_id, 0.0)
                        / max(1.0, self.portfolio.equity())),
                realized_pnl=sleeve.realized_pnl(),
                unrealized_pnl=sleeve.unrealized_pnl(),
                equity=sleeve.equity(),
                n_positions=len(sleeve.open_positions()),
            ))

    # ── one iteration ────────────────────────────────────────────────────

    def run_once(self) -> dict:
        result = {"snapshot_ts": None, "halted": False, "orders_submitted": 0,
                  "fills": 0, "equity": None}
        # 0. kill switch
        if self.state.kill_switch_engaged():
            if not self._killed_handled:
                self.kill.engage("kill switch found engaged")
                self._killed_handled = True
            result["halted"] = True
            self._publish(None)
            return result
        self._killed_handled = False

        try:
            # day rollover
            today = utcnow().date()
            if self._current_day != today:
                self._current_day = today
                self.state.reset_llm_spend()
                # clear the anchor; it is re-set AFTER the broker sync below so
                # "day P&L" measures from the account's real equity, not the
                # config starting_cash (a restart after losses must not
                # instantly trip the daily-loss breaker)
                self.portfolio.day_anchor_equity = None

            # 1. snapshot
            # as_of=None => LIVE mode: the provider stamps the snapshot after
            # collection so freshly-arrived quotes/chains aren't discarded as
            # "future data" (that emptied the option chains on 2026-08-04)
            intraday_syms = sorted({
                u for st in self.strategies
                if getattr(st, "fast_cadence", False)
                for u in getattr(st, "underlyings", [])
            })
            # A fast sleeve's underlyings must have chains fetched even when
            # they are not in the configured options universe — otherwise the
            # sleeve silently sees no contracts and never trades.
            option_unders = sorted(set(self.settings.options_universe)
                                   | set(intraday_syms))
            snapshot = self.provider.build_snapshot(
                self.settings.universe, as_of=None,
                options_underlyings=option_unders,
                intraday_symbols=intraday_syms or None,
            )
            result["snapshot_ts"] = snapshot.as_of.isoformat()
            # keep a sim broker priced off the current snapshot (no-op for Alpaca)
            if hasattr(self.broker, "set_snapshot"):
                self.broker.set_snapshot(snapshot)

            # 2. broker sync + fills
            self._sync_from_broker(snapshot)
            if self.portfolio.day_anchor_equity is None:
                self.portfolio.start_new_day()
            result["fills"] = self._ingest_fills()

            # 2b. EXPIRY GUARD — unconditional, before everything else.
            result["expiry_closed"] = self._close_expiring_options(snapshot)

            # 3. breakers (reconciliation compares the fill-derived local book
            # against broker truth — never the broker-rebuilt mirror)
            trips = self.breakers.check_all(
                self.portfolio, snapshot, getattr(self, "_broker_positions", None),
                local_qty=self._local_qty if self._local_seeded else None)
            if trips:
                reasons = "; ".join(f"{t.name}: {t.reason}" for t in trips)
                self.state.set_halted(True, reasons)
                result["halted"] = True
                fatal = {t.name for t in trips} & {"daily_loss", "drawdown"}
                if fatal and not self._flattened_by_breaker:
                    self.kill.engage(f"auto: {reasons}")
                    self._flattened_by_breaker = True
                self._publish(snapshot)
                self.breakers.record_success()
                result["equity"] = self.portfolio.equity()
                return result
            self.state.set_halted(False)
            self._flattened_by_breaker = False

            # 4. strategy pass (throttled)
            now = utcnow()
            due = (self._last_rebalance is None or
                   (now - self._last_rebalance).total_seconds()
                   >= self.settings.rebalance_interval_sec)
            fast_only = not due and any(
                getattr(st, "fast_cadence", False) for st in self.strategies)
            if due or fast_only:
                if due:
                    self._last_rebalance = now
                total_eq = self.portfolio.equity()
                sleeves_perf = [
                    SleevePerf(sid, self.sleeve_returns[sid])
                    for sid in self.sleeves
                ]
                alloc = self.allocator.allocate(total_eq, sleeves_perf)
                self.sleeve_capital = alloc
                killed_mid_pass = False
                # On a fast-only pass, evaluate ONLY the fast sleeves.
                # Iterating every sleeve here would run the slow sleeves on the
                # poll interval instead of the rebalance interval, silently
                # discarding the throttle that keeps turnover (and friction)
                # down — 2026-08-03 produced 85 equity trades in one session.
                pass_strategies = ([st for st in self.strategies
                                    if getattr(st, "fast_cadence", False)]
                                   if fast_only and not due else self.strategies)
                for strat in pass_strategies:
                    # the kill switch can engage from the dashboard thread while
                    # a strategy pass is running (LLM calls take minutes) —
                    # never submit orders decided before the kill
                    if self.state.kill_switch_engaged():
                        killed_mid_pass = True
                        break
                    sleeve = self.sleeves[strat.strategy_id]
                    ctx = SleeveContext(
                        strategy_id=strat.strategy_id,
                        allocated_capital=self.sleeve_capital.get(
                            strat.strategy_id, 0.0),
                        positions=dict(sleeve.positions),
                        sleeve_equity=sleeve.equity(),
                        data_dir=str(self.settings.data_dir),
                    )
                    try:
                        signals = strat.generate_signals(snapshot, ctx)
                    except Exception as e:  # noqa: BLE001
                        self.breakers.record_error(strat.strategy_id, repr(e))
                        continue
                    for sig in signals:
                        if self.state.kill_switch_engaged():
                            killed_mid_pass = True
                            break
                        order = self._signal_to_order(sig, sleeve, snapshot)
                        if order is None:
                            continue
                        decision = self.risk.check_order(
                            order, self.portfolio, snapshot)
                        if not decision.approved:
                            self.state.log_event(
                                "warning", "risk",
                                f"rejected {order.legs[0].instrument.key} "
                                f"({sig.strategy_id}): {decision.reason}")
                            if "notional" in decision.reason:
                                self.breakers.record_order_bounds_violation(
                                    decision.reason)
                            continue
                        if decision.adjusted_qty is not None:
                            order = single_leg(
                                order.legs[0].instrument, order.legs[0].side,
                                decision.adjusted_qty,
                                strategy_id=order.strategy_id)
                        rationale_id = uuid.uuid4().hex
                        self.state.record_rationale(
                            rationale_id, sig.strategy_id,
                            sig.rationale_payload or {"rationale": sig.rationale})
                        order.rationale_id = rationale_id
                        try:
                            submitted = self.broker.submit_order(order)
                        except BrokerError as e:
                            self.breakers.record_error("broker.submit", str(e))
                            continue
                        if submitted.status == OrderStatus.REJECTED:
                            self.state.log_event(
                                "warning", "broker",
                                f"order rejected: {submitted.reject_reason}")
                            continue
                        key = submitted.broker_order_id or submitted.id
                        self._order_meta[key] = (sig.strategy_id, rationale_id)
                        result["orders_submitted"] += 1
                if killed_mid_pass:
                    self.state.log_event(
                        "critical", "runner",
                        "kill switch engaged mid-iteration — pending signals discarded")
                    self._killed_handled = True
                    result["halted"] = True
                    self._publish(snapshot)
                    result["equity"] = self.portfolio.equity()
                    return result
                result["fills"] += self._ingest_fills()

            # 5. sleeve DAILY-return tracking + publish
            # One sample per TRADING DAY, at the day boundary. The allocator's
            # field is named daily_returns and it annualizes with sqrt(252);
            # appending a sample every loop fed it ~390 intraday returns a day,
            # so Sharpe was computed on 60-second returns and annualized as if
            # daily, and vol targeting (std * sqrt(252)) understated realized
            # vol by ~sqrt(390), quietly disabling itself. This was latent
            # before — sleeve equity only moved on fills — and only became
            # every-loop once sleeve marks started tracking the market.
            today_utc = utcnow().date()
            if self._returns_day is None:
                self._returns_day = today_utc
                self._prev_sleeve_eq = {
                    sid: s.equity() for sid, s in self.sleeves.items()}
            elif today_utc != self._returns_day:
                for sid, sleeve in self.sleeves.items():
                    eq = sleeve.equity()
                    prev = self._prev_sleeve_eq.get(sid)
                    if prev and prev > 0:
                        self.sleeve_returns[sid].append(eq / prev - 1.0)
                        self.sleeve_returns[sid] = self.sleeve_returns[sid][-120:]
                    self._prev_sleeve_eq[sid] = eq
                self._returns_day = today_utc
            self._publish(snapshot)
            self.breakers.record_success()
            result["equity"] = self.portfolio.equity()
            return result
        except Exception as e:  # noqa: BLE001 — the loop must survive anything
            log.exception("run_once iteration failed")
            self.breakers.record_error("runner", repr(e),
                                       transient=_is_transient(e))
            return result

    def run_forever(self, stop_event: threading.Event) -> None:
        self.state.mode = "live"
        self.state.log_event("info", "runner", "paper-trading loop started")
        while not stop_event.is_set():
            self.run_once()
            stop_event.wait(self.settings.poll_interval_sec)
        self.state.log_event("info", "runner", "paper-trading loop stopped")
