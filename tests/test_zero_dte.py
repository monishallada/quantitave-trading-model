"""0DTE sleeve: intraday signals, moneyness selection, brackets, session rules."""
from datetime import date, datetime, time as dtime, timedelta, timezone

import pytest

from quantfund.core.instruments import Option, OptionRight, make_option
from quantfund.core.orders import Fill, OrderSide
from quantfund.core.portfolio import Position
from quantfund.core.snapshot import (
    Bar, MarketSnapshot, OptionChain, OptionQuote, Quote,
)
from quantfund.strategies.base import SignalAction, SleeveContext
from quantfund.strategies.zero_dte import ZeroDTEStrategy

UTC = timezone.utc


def _session_ts(hour=15, minute=0):
    """A UTC time inside the 13:30-20:00 options session."""
    return datetime(2026, 8, 7, hour, minute, tzinfo=UTC)


def _intraday(closes, end_ts, volumes=None):
    n = len(closes)
    vols = volumes or [1000.0] * n
    return tuple(
        Bar(ts=end_ts - timedelta(minutes=n - 1 - i), open=c, high=c * 1.001,
            low=c * 0.999, close=c, volume=vols[i])
        for i, c in enumerate(closes)
    )


def _chain(spot, ts, expiry=None, strikes=None):
    expiry = expiry or ts.date()          # 0DTE by default
    strikes = strikes or [round(spot + k) for k in (-4, -2, -1, 0, 1, 2, 4)]
    qs = []
    for k in strikes:
        for right in (OptionRight.CALL, OptionRight.PUT):
            inst = make_option("SPY", expiry, right, float(k))
            prem = 1.50
            qs.append(OptionQuote(
                instrument=inst,
                quote=Quote(ts=ts, bid=prem - 0.01, ask=prem + 0.01),
                iv=None, greeks=None))       # 0DTE feed has NO greeks
    return OptionChain(underlying="SPY", ts=ts, underlying_price=spot,
                       quotes=tuple(qs))


def _snap(closes, ts=None, volumes=None, spot=None, expiry=None):
    ts = ts or _session_ts()
    spot = spot if spot is not None else closes[-1]
    return MarketSnapshot(
        as_of=ts,
        intraday_bars={"SPY": _intraday(closes, ts, volumes)},
        quotes={"SPY": Quote(ts=ts, bid=spot - 0.01, ask=spot + 0.01)},
        option_chains={"SPY": _chain(spot, ts, expiry)},
    )


def ctx(positions=None, capital=30_000.0):
    return SleeveContext(strategy_id="zero_dte", allocated_capital=capital,
                         positions=positions or {}, sleeve_equity=capital)


def rising(n=40, start=770.0, step=0.12):
    return [start + step * i for i in range(n)]


def falling(n=40, start=770.0, step=0.12):
    return [start - step * i for i in range(n)]


def flat(n=40, px=770.0):
    return [px] * n


class TestEntries:
    def test_buys_0dte_call_on_up_signal(self):
        vols = [1000.0] * 35 + [4000.0] * 5          # volume surge
        s = ZeroDTEStrategy(underlyings=["SPY"])
        sigs = s.generate_signals(_snap(rising(), volumes=vols), ctx())
        opens = [x for x in sigs if x.action == SignalAction.OPEN_LONG]
        assert len(opens) == 1
        inst = opens[0].instrument
        assert isinstance(inst, Option) and inst.right == OptionRight.CALL
        assert inst.expiration == _session_ts().date()      # genuinely 0DTE
        assert opens[0].target_weight > 0
        assert "max loss = premium" in opens[0].rationale

    def test_buys_0dte_put_on_down_signal(self):
        vols = [1000.0] * 35 + [4000.0] * 5
        s = ZeroDTEStrategy(underlyings=["SPY"])
        sigs = s.generate_signals(_snap(falling(), volumes=vols), ctx())
        opens = [x for x in sigs if x.action == SignalAction.OPEN_LONG]
        assert len(opens) == 1
        assert opens[0].instrument.right == OptionRight.PUT

    def test_no_trade_without_volume_confirmation(self):
        s = ZeroDTEStrategy(underlyings=["SPY"])
        sigs = s.generate_signals(_snap(rising()), ctx())   # flat volume
        assert not any(x.action == SignalAction.OPEN_LONG for x in sigs)

    def test_no_trade_in_chop(self):
        vols = [1000.0] * 35 + [4000.0] * 5
        s = ZeroDTEStrategy(underlyings=["SPY"])
        sigs = s.generate_signals(_snap(flat(), volumes=vols), ctx())
        assert not any(x.action == SignalAction.OPEN_LONG for x in sigs)

    def test_wide_spreads_rejected(self):
        ts = _session_ts()
        spot = 770.0
        wide = OptionChain(
            underlying="SPY", ts=ts, underlying_price=spot,
            quotes=tuple(
                OptionQuote(instrument=make_option("SPY", ts.date(), r, float(k)),
                            quote=Quote(ts=ts, bid=1.00, ask=1.60))  # 46% spread
                for k in (768, 769, 770, 771, 772)
                for r in (OptionRight.CALL, OptionRight.PUT)))
        vols = [1000.0] * 35 + [4000.0] * 5
        snap = MarketSnapshot(as_of=ts,
                              intraday_bars={"SPY": _intraday(rising(), ts, vols)},
                              quotes={"SPY": Quote(ts=ts, bid=769.9, ask=770.1)},
                              option_chains={"SPY": wide})
        s = ZeroDTEStrategy(underlyings=["SPY"])
        assert not any(x.action == SignalAction.OPEN_LONG
                       for x in s.generate_signals(snap, ctx()))

    def test_never_selects_a_later_expiry(self):
        vols = [1000.0] * 35 + [4000.0] * 5
        ts = _session_ts()
        snap = _snap(rising(), ts=ts, volumes=vols,
                     expiry=ts.date() + timedelta(days=7))   # no 0DTE available
        s = ZeroDTEStrategy(underlyings=["SPY"])
        assert not any(x.action == SignalAction.OPEN_LONG
                       for x in s.generate_signals(snap, ctx()))


class TestSessionRules:
    def test_no_entries_late_in_session(self):
        vols = [1000.0] * 35 + [4000.0] * 5
        s = ZeroDTEStrategy(underlyings=["SPY"])
        late = _snap(rising(), ts=_session_ts(19, 0), volumes=vols)  # 15:00 ET
        assert not any(x.action == SignalAction.OPEN_LONG
                       for x in s.generate_signals(late, ctx()))

    def test_daily_entry_budget(self):
        vols = [1000.0] * 35 + [4000.0] * 5
        s = ZeroDTEStrategy(underlyings=["SPY"], max_entries_per_day=1,
                            max_concurrent=5)
        snap = _snap(rising(), volumes=vols)
        first = s.generate_signals(snap, ctx())
        second = s.generate_signals(snap, ctx())
        assert any(x.action == SignalAction.OPEN_LONG for x in first)
        assert not any(x.action == SignalAction.OPEN_LONG for x in second)


class TestBrackets:
    def _held(self, entry=1.50, mark=1.50, ts=None):
        ts = ts or _session_ts()
        inst = make_option("SPY", ts.date(), OptionRight.CALL, 770.0)
        pos = Position(instrument=inst)
        pos.apply_fill(Fill(order_id="x", instrument=inst, side=OrderSide.BUY,
                            qty=2, price=entry, ts=ts))
        pos.mark = mark
        return inst, pos

    def test_take_profit(self):
        inst, pos = self._held(entry=1.50, mark=2.50)
        s = ZeroDTEStrategy(underlyings=["SPY"])
        closes = [x for x in s.generate_signals(_snap(flat()), ctx({inst.key: pos}))
                  if x.action == SignalAction.CLOSE]
        assert closes and "take profit" in closes[0].rationale

    def test_stop_loss(self):
        inst, pos = self._held(entry=1.50, mark=0.90)
        s = ZeroDTEStrategy(underlyings=["SPY"])
        closes = [x for x in s.generate_signals(_snap(flat()), ctx({inst.key: pos}))
                  if x.action == SignalAction.CLOSE]
        assert closes and "stop loss" in closes[0].rationale

    def test_forced_flat_before_expiry(self):
        """A 0DTE contract must NEVER be held into the close."""
        ts = _session_ts(19, 45)          # 15:45 ET, past force-flat
        inst, pos = self._held(entry=1.50, mark=1.55, ts=ts)
        s = ZeroDTEStrategy(underlyings=["SPY"])
        snap = _snap(flat(), ts=ts)
        closes = [x for x in s.generate_signals(snap, ctx({inst.key: pos}))
                  if x.action == SignalAction.CLOSE]
        assert closes and "forced flat" in closes[0].rationale


def test_sleeve_is_marked_unvalidated():
    s = ZeroDTEStrategy()
    assert "UNVALIDATED" in s.name
    assert s.fast_cadence is True
    import quantfund.strategies.zero_dte as mod
    assert "cannot" in (mod.__doc__ or "").lower()
    assert "UNVALIDATED" in (ZeroDTEStrategy.__doc__ or "")


class TestVolumeBaseline:
    """Regression: the surge baseline must not include the opening burst.

    Intraday equity volume is U-shaped. The original implementation compared
    the last 5 minutes to the MEAN of the whole session window, so once the
    opening burst was in the window the ratio sat below 1.0 for most of the
    day and the sleeve could not fire. Observed live 2026-08-07: SPY 0.81,
    QQQ 0.76, IWM 0.34 at 10:39 ET on a normal tape.
    """

    @staticmethod
    def _u_shaped(n=90, opening_burst=25.0, midday=1000.0):
        """First 10 minutes carry 25x the volume of the midday tape."""
        return [midday * opening_burst] * 10 + [midday] * (n - 10)

    def test_opening_burst_does_not_suppress_midday_signal(self):
        vols = self._u_shaped()
        vols[-5:] = [1400.0] * 5           # a genuine 1.4x midday surge
        s = ZeroDTEStrategy(underlyings=["SPY"])
        _, detail = s._intraday_signal(
            "SPY", _snap(rising(90), volumes=vols))
        assert detail["volume_surge"] >= 1.3, (
            f"opening burst leaked into the baseline: surge={detail['volume_surge']}")

    def test_quiet_tape_still_produces_no_surge(self):
        """The fix must not turn the filter into a no-op."""
        vols = self._u_shaped()             # midday flat, no recent surge
        s = ZeroDTEStrategy(underlyings=["SPY"])
        _, detail = s._intraday_signal(
            "SPY", _snap(rising(90), volumes=vols))
        assert detail["volume_surge"] < 1.3
        assert detail["reason"] == "no volume confirmation"

    def test_baseline_excludes_the_recent_bars_it_measures(self):
        """A surge must not inflate its own baseline and cancel itself out."""
        vols = [1000.0] * 85 + [5000.0] * 5
        s = ZeroDTEStrategy(underlyings=["SPY"])
        _, detail = s._intraday_signal(
            "SPY", _snap(rising(90), volumes=vols))
        assert detail["volume_surge"] == pytest.approx(5.0, rel=0.01)
