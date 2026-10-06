import numpy as np
import pandas as pd
from pairs import Config, select_pairs, signal_positions, backtest_pair, run_pipeline, portfolio_pnl


def synthetic(n=1800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    walks = {c: np.cumsum(rng.normal(0, 0.015, n)) + 5 for c in "BCDE"}
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.9 * s[t - 1] + rng.normal(0, 0.02)       # mean-reverting spread
    df = pd.DataFrame(walks, index=idx)
    df["A"] = 1.2 * df["B"] + s
    return np.exp(df)                                      # price levels


CFG = Config(train_end="2022-06-30")


def test_selects_planted_pair():
    logp = np.log(synthetic()).loc[:CFG.train_end]
    t = select_pairs(logp, CFG)
    sel = t[t.selected]
    assert len(sel) >= 1
    assert {"A", "B"} == set(sel.iloc[0][["a", "b"]])
    r = sel.iloc[0]
    expected = 1.2 if r.a == "A" else 1 / 1.2
    assert abs(r.beta - expected) < 0.1


def test_signals_are_causal():
    """Position at day t must not change if future data is removed."""
    px = synthetic(); logp = np.log(px)
    full = backtest_pair(logp, "A", "B", 0.0, 1.2, CFG, "2022-07-01", logp.index[-1])
    cut = logp.index[-200]
    part = backtest_pair(logp.loc[:cut], "A", "B", 0.0, 1.2, CFG, "2022-07-01", cut)
    common = part.index[:-1]                               # last row of `part` includes forced liquidation cost only
    assert (full.loc[common, "pos"] == part.loc[common, "pos"]).all()


def test_costs_reduce_pnl_and_pipeline_runs():
    res = run_pipeline(synthetic(), CFG)
    assert res["oos_frames"]
    assert portfolio_pnl(res["oos_frames"], 20).sum() < portfolio_pnl(res["oos_frames"], 0).sum()


def test_stop_blocks_reentry_beyond_stop():
    z = pd.Series([0, 2.5, 4.5, 4.5, 3.0, 0.2])
    pos = signal_positions(z, CFG).tolist()
    assert pos == [0, -1, 0, 0, -1, 0]
