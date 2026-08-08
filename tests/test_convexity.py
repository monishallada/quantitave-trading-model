"""Convexity sleeve v2: gated 2-way long-options entries, regime rules, exits."""
from datetime import timedelta

from quantfund.core.config import RiskLimits, Settings
from quantfund.core.instruments import Option, OptionRight, make_option
from quantfund.core.orders import Fill, OrderSide
from quantfund.core.portfolio import Position
from quantfund.core.snapshot import MarketSnapshot, RegimeState
from quantfund.strategies.base import SignalAction, SleeveContext
from quantfund.strategies.convexity import ConvexMomentumStrategy

from conftest import T0, make_bars, make_chain, make_quote

N = 140  # enough for the 126d slow-momentum gate


def trending_closes(n=N, start=100.0, daily=0.004):
    return [start * (1 + daily) ** i for i in range(n)]


def snap(closes_by_symbol, trend="sideways", vol="normal", chain_spots=None):
    bars = {s: make_bars(c, end_ts=T0) for s, c in closes_by_symbol.items()}
    quotes = {s: make_quote(c[-1], ts=T0) for s, c in closes_by_symbol.items()}
    chains = {}
    for s, c in closes_by_symbol.items():
        spot = (chain_spots or {}).get(s, c[-1])
        chains[s] = make_chain(s, spot=spot, ts=T0,
                               expiry=(T0 + timedelta(days=21)).date())
    return MarketSnapshot(
        as_of=T0, bars=bars, quotes=quotes, option_chains=chains,
        regime=RegimeState(as_of=T0, vol_regime=vol, trend_regime=trend),
    )


def ctx_for(positions=None, capital=50_000.0):
    return SleeveContext(strategy_id="convexity", allocated_capital=capital,
                         positions=positions or {}, sleeve_equity=capital)


class TestEntries:
    def test_calls_need_slow_and_fast_momentum(self):
        # UP passes both gates (126d ~+75%, 20d ~+8%); SLOWFADE has a strong
        # 20d bounce inside a longer decline → 126d gate rejects it
        decline = trending_closes(N - 21, daily=-0.0015)
        slowfade = decline + [decline[-1] * (1 + 0.006) ** i
                              for i in range(1, 22)]
        s = snap({"UP": trending_closes(daily=0.004), "SLOWFADE": slowfade})
        signals = ConvexMomentumStrategy().generate_signals(s, ctx_for())
        opens = [x for x in signals if x.action == SignalAction.OPEN_LONG]
        under = {x.rationale_payload["underlying"] for x in opens}
        assert "UP" in under
        assert "SLOWFADE" not in under
        for x in opens:
            assert x.instrument.right == OptionRight.CALL
            assert x.target_weight > 0
            assert "max loss = premium" in x.rationale

    def test_no_puts_in_calm_bull_market(self):
        """2023-25 lesson: buying puts in a calm uptrend is the bleed. Gated."""
        s = snap({"DOWN": trending_closes(daily=-0.005)},
                 trend="up", vol="normal")
        signals = ConvexMomentumStrategy().generate_signals(s, ctx_for())
        assert not any(x.action == SignalAction.OPEN_LONG for x in signals)

    def test_puts_fire_in_downtrend_regime(self):
        s = snap({"DOWN": trending_closes(daily=-0.005)},
                 trend="down", vol="normal")
        signals = ConvexMomentumStrategy().generate_signals(s, ctx_for())
        opens = [x for x in signals if x.action == SignalAction.OPEN_LONG]
        assert len(opens) == 1
        assert opens[0].instrument.right == OptionRight.PUT

    def test_puts_fire_in_high_vol_regime(self):
        s = snap({"DOWN": trending_closes(daily=-0.005)},
                 trend="sideways", vol="high")
        signals = ConvexMomentumStrategy().generate_signals(s, ctx_for())
        opens = [x for x in signals if x.action == SignalAction.OPEN_LONG]
        assert len(opens) == 1
        assert opens[0].instrument.right == OptionRight.PUT

    def test_no_calls_in_downtrend_regime(self):
        s = snap({"UP": trending_closes(daily=0.004),
                  "D1": trending_closes(daily=-0.004),
                  "D2": trending_closes(daily=-0.005)}, trend="down")
        signals = ConvexMomentumStrategy().generate_signals(s, ctx_for())
        opens = [x for x in signals if x.action == SignalAction.OPEN_LONG]
        assert opens, "downtrend with falling names should buy puts"
        assert all(x.instrument.right == OptionRight.PUT for x in opens)

    def test_weak_moves_skipped(self):
        # +1%/20d is below the 5% confirmation gate even with strong 126d…
        slow_only = trending_closes(N - 21, daily=0.004)
        slow_only += [slow_only[-1] * (1 + 0.0005) ** i for i in range(1, 22)]
        s = snap({"WEAK": slow_only})
        signals = ConvexMomentumStrategy().generate_signals(s, ctx_for())
        assert not any(x.action == SignalAction.OPEN_LONG for x in signals)

    def test_dte_window_14_35(self):
        # chain with only a 7-day expiry → outside the 14-35 window → no trade
        bars = {"UP": make_bars(trending_closes(daily=0.004), end_ts=T0)}
        chain = make_chain("UP", spot=trending_closes(daily=0.004)[-1], ts=T0,
                           expiry=(T0 + timedelta(days=7)).date())
        s = MarketSnapshot(as_of=T0, bars=bars,
                           quotes={"UP": make_quote(100.0, ts=T0)},
                           option_chains={"UP": chain})
        signals = ConvexMomentumStrategy().generate_signals(s, ctx_for())
        assert not any(x.action == SignalAction.OPEN_LONG for x in signals)


class TestExits:
    def _pos(self, entry=3.0, mark=3.0, expiry_days=21, right=OptionRight.CALL):
        inst = make_option("UP", (T0 + timedelta(days=expiry_days)).date(),
                           right, 105.0)
        p = Position(instrument=inst)
        p.apply_fill(Fill(order_id="x", instrument=inst, side=OrderSide.BUY,
                          qty=5, price=entry, ts=T0))
        p.mark = mark
        return inst, p

    def _signals_for(self, pos_map, closes=None):
        s = snap({"UP": closes or trending_closes(daily=0.004)})
        return ConvexMomentumStrategy().generate_signals(s, ctx_for(pos_map))

    def test_take_profit_at_3x(self):
        inst, p = self._pos(entry=3.0, mark=9.5)
        closes = [x for x in self._signals_for({inst.key: p})
                  if x.action == SignalAction.CLOSE]
        assert closes and "take profit" in closes[0].rationale

    def test_premium_stop_at_04x(self):
        inst, p = self._pos(entry=3.0, mark=1.0)
        closes = [x for x in self._signals_for({inst.key: p})
                  if x.action == SignalAction.CLOSE]
        assert closes and "premium stop" in closes[0].rationale

    def test_time_exit(self):
        inst, p = self._pos(entry=3.0, mark=3.0, expiry_days=2)
        closes = [x for x in self._signals_for({inst.key: p})
                  if x.action == SignalAction.CLOSE]
        assert closes and "time exit" in closes[0].rationale

    def test_momentum_flip_exit(self):
        # long a CALL while the underlying's 20d momentum is now negative
        inst, p = self._pos(entry=3.0, mark=3.0)
        closes = [x for x in self._signals_for(
                      {inst.key: p}, closes=trending_closes(daily=-0.005))
                  if x.action == SignalAction.CLOSE]
        assert closes and "momentum flipped" in closes[0].rationale


class TestExplosiveProfile:
    def test_env_switch(self, tmp_path, monkeypatch):
        from quantfund.core.config import load_settings
        monkeypatch.setenv("QF_RISK_PROFILE", "explosive")
        monkeypatch.setenv("QF_DATA_DIR", str(tmp_path))
        s = load_settings(env_file=None)
        assert s.risk_profile == "explosive"
        assert s.risk.daily_loss_halt_pct == 0.25
        assert s.risk.allow_short_equity is True
        assert s.risk.allow_naked_short_options is False  # stays banned
        assert s.risk.max_net_delta_pct == 40.0

    def test_default_stays_conservative(self, tmp_path, monkeypatch):
        from quantfund.core.config import load_settings
        monkeypatch.delenv("QF_RISK_PROFILE", raising=False)
        monkeypatch.setenv("QF_DATA_DIR", str(tmp_path))
        s = load_settings(env_file=None)
        assert s.risk_profile == "conservative"
        assert s.risk.daily_loss_halt_pct == RiskLimits().daily_loss_halt_pct


class TestCrashConvexityMode:
    """calls_enabled=False: puts-only crash hedge (momentum's deep-ITM calls
    carry the bullish side)."""

    def test_no_calls_when_disabled(self):
        s = snap({"UP": trending_closes(daily=0.004)}, trend="up")
        strat = ConvexMomentumStrategy(calls_enabled=False)
        signals = strat.generate_signals(s, ctx_for())
        assert not any(x.action == SignalAction.OPEN_LONG for x in signals)

    def test_puts_still_fire_in_stress(self):
        s = snap({"DOWN": trending_closes(daily=-0.005)}, trend="down")
        strat = ConvexMomentumStrategy(calls_enabled=False)
        opens = [x for x in strat.generate_signals(s, ctx_for())
                 if x.action == SignalAction.OPEN_LONG]
        assert len(opens) == 1
        assert opens[0].instrument.right == OptionRight.PUT


def test_explosive_per_name_cap_admits_one_index_contract():
    """A single deep-ITM index call (~50% of a $100k account in delta) must
    fit under the per-name cap, or the options sleeve sits in cash."""
    lim = RiskLimits.explosive()
    equity = 100_000.0
    one_spy_contract_delta = 0.65 * 100 * 771.0   # ~$50k
    assert one_spy_contract_delta < lim.max_position_pct * equity
    # but two of the same name must NOT fit
    assert 2 * one_spy_contract_delta > lim.max_position_pct * equity
