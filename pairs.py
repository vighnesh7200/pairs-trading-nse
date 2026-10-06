"""Cointegration-based pairs trading: selection, signals, backtest, metrics.

Design rules (to avoid look-ahead):
  * pairs, hedge ratios (alpha, beta) are estimated on the TRAIN window only and frozen
  * z-score uses a rolling window of past spread values only
  * signal at close of day t -> position held over day t+1 (shift by one day)
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint


@dataclass
class Config:
    train_end: str = "2022-12-31"
    z_window: int = 60          # rolling window (days) for spread mean/std
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 4.0         # stop-loss: close and do not re-enter beyond this
    fdr_alpha: float = 0.10     # Benjamini-Hochberg false discovery rate on EG p-values
    min_half_life: float = 2.0  # days
    max_half_life: float = 60.0
    max_pairs: int = 5
    cost_bps: float = 5.0       # per side, on traded notional: brokerage, taxes, exchange fees
    slip_bps: float = 5.0       # per side, slippage
    ann: int = 252

    @property
    def total_bps(self):
        return self.cost_bps + self.slip_bps


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life (days) from an AR(1) fit on the spread."""
    lag = spread.shift(1).dropna().values
    ds = spread.diff().dropna().values
    b = np.polyfit(lag, ds, 1)[0]
    return float(-np.log(2) / np.log(1 + b)) if -1 < b < 0 else np.inf


def select_pairs(train_logp: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Engle-Granger test on every pair (train data only), FDR-corrected."""
    rows = []
    for a, b in combinations(train_logp.columns, 2):
        pval = coint(train_logp[a], train_logp[b], trend="c")[1]
        ols = sm.OLS(train_logp[a], sm.add_constant(train_logp[b])).fit()
        alpha, beta = float(ols.params.iloc[0]), float(ols.params.iloc[1])
        spread = train_logp[a] - beta * train_logp[b] - alpha
        rows.append(dict(a=a, b=b, pvalue=pval, alpha=alpha, beta=beta,
                         half_life=half_life(spread)))
    df = pd.DataFrame(rows)
    reject, p_adj, _, _ = multipletests(df["pvalue"], alpha=cfg.fdr_alpha, method="fdr_bh")
    df["p_adj"] = p_adj
    ok = (reject & (df["beta"] > 0)
          & df["half_life"].between(cfg.min_half_life, cfg.max_half_life))
    df["selected"] = False
    top = df[ok].sort_values("p_adj").head(cfg.max_pairs).index
    df.loc[top, "selected"] = True
    return df.sort_values("pvalue").reset_index(drop=True)


def signal_positions(z: pd.Series, cfg: Config) -> pd.Series:
    """+1 = long spread (long A, short B), -1 = short spread, 0 = flat."""
    out, cur = np.zeros(len(z)), 0
    for i, zi in enumerate(z.values):
        if np.isnan(zi):
            cur = 0
        elif cur == 0:
            if cfg.z_entry < zi < cfg.z_stop:
                cur = -1
            elif -cfg.z_stop < zi < -cfg.z_entry:
                cur = 1
        elif abs(zi) < cfg.z_exit or abs(zi) > cfg.z_stop:
            cur = 0
        out[i] = cur
    return pd.Series(out, index=z.index)


def backtest_pair(logp, a, b, alpha, beta, cfg, start, end):
    """Backtest one pair over [start, end] using frozen (alpha, beta). Returns daily frame."""
    spread = logp[a] - beta * logp[b] - alpha
    z = (spread - spread.rolling(cfg.z_window).mean()) / spread.rolling(cfg.z_window).std()
    idx = logp.loc[start:end].index
    ret = logp.diff().loc[idx]
    pos = signal_positions(z.loc[idx], cfg).shift(1).fillna(0.0)    # trade at close, hold next day
    gross = pos * (ret[a] - beta * ret[b]) / (1 + abs(beta))        # return on gross notional
    turnover = pos.diff().abs().fillna(pos.abs())                   # fraction of gross notional traded
    turnover.iloc[-1] += abs(pos.iloc[-1])                          # liquidate at the end
    return pd.DataFrame({"z": z.loc[idx], "pos": pos, "gross": gross, "turnover": turnover})


def n_trades(df) -> int:
    return int(((df["pos"] != 0) & (df["pos"].shift(1).fillna(0) == 0)).sum())


def portfolio_pnl(frames: dict, bps: float) -> pd.Series:
    """Equal-capital portfolio of pairs; net of costs at `bps` per side on traded notional."""
    net = [f["gross"] - f["turnover"] * bps / 1e4 for f in frames.values()]
    return pd.concat(net, axis=1).mean(axis=1)


def metrics(pnl: pd.Series, ann: int = 252) -> dict:
    eq = (1 + pnl).cumprod()
    sd = pnl.std()
    return {
        "ann_return": pnl.mean() * ann,
        "ann_vol": sd * np.sqrt(ann),
        "sharpe": pnl.mean() / sd * np.sqrt(ann) if sd > 0 else np.nan,
        "max_drawdown": (eq / eq.cummax() - 1).min(),
        "total_return": eq.iloc[-1] - 1,
    }


def run_pipeline(prices: pd.DataFrame, cfg: Config) -> dict:
    logp = np.log(prices.dropna(how="any"))
    train_end = pd.Timestamp(cfg.train_end)
    train = logp.loc[:train_end]
    test_idx = logp.index[logp.index > train_end]
    table = select_pairs(train, cfg)
    sel = table[table["selected"]]
    is_frames, oos_frames = {}, {}
    for _, r in sel.iterrows():
        key = f"{r.a}/{r.b}"
        is_frames[key] = backtest_pair(logp, r.a, r.b, r.alpha, r.beta, cfg,
                                       train.index[0], train.index[-1])
        oos_frames[key] = backtest_pair(logp, r.a, r.b, r.alpha, r.beta, cfg,
                                        test_idx[0], test_idx[-1])
    return dict(table=table, selected=sel, is_frames=is_frames, oos_frames=oos_frames,
                n_pairs_tested=len(table), train_range=(train.index[0], train.index[-1]),
                test_range=(test_idx[0], test_idx[-1]))
