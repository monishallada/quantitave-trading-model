"""CapitalAllocator: caps, floors, correlation penalty, determinism."""
import math

from quantfund.allocation.allocator import CapitalAllocator, SleevePerf
from quantfund.core.config import RiskLimits


def alloc(sleeves, equity=100_000, **kw):
    return CapitalAllocator(RiskLimits(), **kw).allocate(equity, sleeves)


def test_no_history_equal_weights():
    out = alloc([SleevePerf("a", []), SleevePerf("b", []), SleevePerf("c", [])])
    vals = list(out.values())
    assert all(abs(v - vals[0]) < 1.0 for v in vals)
    assert sum(vals) <= 100_000 + 1e-6


def test_caps_and_floors_respected():
    good = [0.01] * 60           # steady winner
    bad = [-0.01] * 60           # steady loser
    out = alloc([SleevePerf("good", good), SleevePerf("bad", bad)])
    total = 100_000
    assert out["good"] <= RiskLimits().max_strategy_pct * total + 1e-6
    assert out["bad"] >= 0.05 * total - 1e-6   # floor
    assert out["good"] > out["bad"]


def test_sum_never_exceeds_equity():
    import random
    random.seed(1)
    sleeves = [SleevePerf(f"s{i}", [random.gauss(0.001, 0.01) for _ in range(60)])
               for i in range(4)]
    out = alloc(sleeves)
    assert sum(out.values()) <= 100_000 * (1 + 1e-9)


def test_correlation_penalty():
    import random
    base = [math.sin(i / 3) * 0.01 + 0.002 for i in range(60)]
    # a fixed permutation has EXACTLY the same mean/std (same Sharpe) but is
    # decorrelated from the original ordering
    solo = list(base)
    random.Random(123).shuffle(solo)
    out = alloc([SleevePerf("twin1", base), SleevePerf("twin2", list(base)),
                 SleevePerf("solo", solo)])
    assert out["solo"] > out["twin1"]
    assert out["solo"] > out["twin2"]


def test_deterministic():
    sleeves = [SleevePerf("a", [0.001 * i for i in range(30)]),
               SleevePerf("b", [0.002] * 30)]
    assert alloc(sleeves) == alloc(sleeves)


def test_single_sleeve():
    out = alloc([SleevePerf("only", [0.01] * 60)])
    assert 0 < out["only"] <= RiskLimits().max_strategy_pct * 100_000 + 1e-6


def test_vol_targeting_scales_down_hot_portfolios():
    """Realized portfolio vol far above the 12% target → deployment shrinks
    (floored at 30%); calm portfolios deploy fully."""
    hot = [0.03 if i % 2 == 0 else -0.03 for i in range(60)]    # ~48% ann vol
    calm = [0.0003 if i % 2 == 0 else -0.0003 for i in range(60)]  # ~0.5% ann
    hot_alloc = alloc([SleevePerf("a", hot), SleevePerf("b", list(hot))])
    calm_alloc = alloc([SleevePerf("a", calm), SleevePerf("b", list(calm))])
    assert sum(hot_alloc.values()) < 0.5 * sum(calm_alloc.values())
    assert sum(hot_alloc.values()) >= 0.30 * 0.8 * 100_000 * 0.9  # floor holds


def test_base_weights_set_the_no_history_split():
    """Equal-weighting a 4-sleeve book caps the main options engine at 25% of
    equity; base weights make a ~50%-in-options target reachable."""
    sleeves = [SleevePerf(k, []) for k in
               ("momentum", "put_income", "momentum_equity", "convexity")]
    a = CapitalAllocator(RiskLimits.explosive(), lookback=30, min_weight=0.08,
                         base_weights={"momentum": 0.50, "put_income": 0.22,
                                       "momentum_equity": 0.18, "convexity": 0.10})
    out = a.allocate(100_000, sleeves)
    assert out["momentum"] > 2 * out["convexity"]
    assert out["momentum"] >= 40_000        # the options engine is the core
    assert sum(out.values()) <= 100_000 + 1e-6

def test_base_weights_still_yield_to_performance():
    """A base-weighted sleeve that loses money must still lose allocation."""
    winner = [0.01] * 60
    loser = [-0.01] * 60
    a = CapitalAllocator(RiskLimits.explosive(), lookback=30, min_weight=0.05,
                         base_weights={"momentum": 0.80, "put_income": 0.20})
    out = a.allocate(100_000, [SleevePerf("momentum", loser),
                               SleevePerf("put_income", winner)])
    assert out["put_income"] > out["momentum"]
