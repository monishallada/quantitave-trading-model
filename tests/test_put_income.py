"""Put-income (VRP) sleeve: cash-secured entries, yield gate, regime gate,
exit rules, affordability."""
from datetime import timedelta

from quantfund.core.instruments import Option, OptionRight
from quantfund.core.orders import Fill, OrderSide
from quantfund.core.portfolio import Position
from quantfund.core.snapshot import MarketSnapshot, RegimeState
from quantfund.strategies.base import SignalAction, SleeveContext
from quantfund.strategies.put_income import PutIncomeStrategy

from conftest import T0, make_bars, make_chain, make_quote


def snap_with_chain(spot=200.0, regime_vol="normal", underlying="AAPL"):
    chain = make_chain(underlying, spot=spot)
    closes = [spot] * 60
    return MarketSnapshot(
        as_of=T0,
        bars={underlying: make_bars(closes, end_ts=T0)},
        quotes={underlying: make_quote(spot, ts=T0)},
        option_chains={underlying: chain},
        regime=RegimeState(as_of=T0, vol_regime=regime_vol, trend_regime="up"),
    )


def ctx_for(positions=None, capital=50_000.0):
    return SleeveContext(strategy_id="put_income", allocated_capital=capital,
                         positions=positions or {}, sleeve_equity=capital)


def strat(**kw):
    kw.setdefault("min_annualized_yield", 0.02)  # conftest premiums are lean
    kw.setdefault("preferred", ["AAPL"])
    return PutIncomeStrategy(**kw)


class TestEntries:
    def test_sells_cash_secured_put(self):
        signals = strat().generate_signals(snap_with_chain(), ctx_for())
        assert len(signals) == 1
        sig = signals[0]
        assert sig.action == SignalAction.OPEN_SHORT
        assert sig.target_weight < 0
        assert isinstance(sig.instrument, Option)
        assert sig.instrument.right == OptionRight.PUT
        p = sig.rationale_payload
        # ~30-delta pick on the conftest chain is the 190 strike
        assert p["strike"] == 190.0
        # secured cash within 90% of sleeve capital
        assert p["cash_secured"] <= 0.9 * 50_000
        assert p["contracts"] >= 1
        assert "cash-secured" in sig.rationale

    def test_no_new_entries_in_high_vol(self):
        signals = strat().generate_signals(
            snap_with_chain(regime_vol="high"), ctx_for())
        assert signals == []

    def test_yield_gate_skips_cheap_vol(self):
        # default 8% annualized yield floor vs the lean conftest premiums
        s = PutIncomeStrategy(preferred=["AAPL"])
        assert s.generate_signals(snap_with_chain(), ctx_for()) == []

    def test_unaffordable_strikes_skipped(self):
        # spot 2000 → strike ~1900 → $190k secured >> 90% of $50k sleeve
        signals = strat().generate_signals(
            snap_with_chain(spot=2000.0), ctx_for())
        assert signals == []

    def test_no_double_entry_on_same_underlying(self):
        snap = snap_with_chain()
        chain = snap.get_option_chain("AAPL")
        put = next(q.instrument for q in chain.quotes
                   if q.instrument.right == OptionRight.PUT
                   and q.instrument.strike == 190.0)
        pos = Position(instrument=put)
        pos.apply_fill(Fill(order_id="x", instrument=put, side=OrderSide.SELL,
                            qty=1, price=1.0, ts=T0))
        pos.mark = 1.0
        signals = strat().generate_signals(snap, ctx_for({put.key: pos}))
        assert not any(s.action == SignalAction.OPEN_SHORT for s in signals)


class TestExits:
    def _short_put_pos(self, snap, entry=2.0, mark=None, strike=190.0):
        chain = snap.get_option_chain("AAPL")
        put = next(q.instrument for q in chain.quotes
                   if q.instrument.right == OptionRight.PUT
                   and q.instrument.strike == strike)
        pos = Position(instrument=put)
        pos.apply_fill(Fill(order_id="x", instrument=put, side=OrderSide.SELL,
                            qty=2, price=entry, ts=T0))
        pos.mark = mark if mark is not None else entry
        return put, pos

    def test_profit_take(self):
        snap = snap_with_chain()
        put, pos = self._short_put_pos(snap, entry=2.0, mark=0.9)
        signals = strat().generate_signals(snap, ctx_for({put.key: pos}))
        closes = [s for s in signals if s.action == SignalAction.CLOSE]
        assert len(closes) == 1
        assert "profit take" in closes[0].rationale

    def test_loss_stop(self):
        snap = snap_with_chain()
        put, pos = self._short_put_pos(snap, entry=2.0, mark=5.5)
        signals = strat().generate_signals(snap, ctx_for({put.key: pos}))
        closes = [s for s in signals if s.action == SignalAction.CLOSE]
        assert len(closes) == 1
        assert "loss stop" in closes[0].rationale

    def test_time_exit_near_expiry(self):
        snap = snap_with_chain()
        # hand-build a put expiring in 5 days, held short at a steady mark
        from quantfund.core.instruments import make_option
        near = make_option("AAPL", (T0 + timedelta(days=5)).date(),
                           OptionRight.PUT, 190.0)
        pos = Position(instrument=near)
        pos.apply_fill(Fill(order_id="x", instrument=near, side=OrderSide.SELL,
                            qty=1, price=2.0, ts=T0))
        pos.mark = 1.5
        signals = strat().generate_signals(snap, ctx_for({near.key: pos}))
        closes = [s for s in signals if s.action == SignalAction.CLOSE]
        assert len(closes) == 1
        assert "time exit" in closes[0].rationale
