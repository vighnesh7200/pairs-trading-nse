"""Download NSE bank prices, run the pairs-trading backtest, save results + figures."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pairs import Config, run_pipeline, portfolio_pnl, metrics, n_trades

TICKERS = ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
           "INDUSINDBK", "BANKBARODA", "PNB", "FEDERALBNK", "IDFCFIRSTB"]
START = "2018-01-01"
CACHE = "data/prices.csv"


def load_prices(refresh=False) -> pd.DataFrame:
    if os.path.exists(CACHE) and not refresh:
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)
    import yfinance as yf
    raw = yf.download([t + ".NS" for t in TICKERS], start=START, auto_adjust=True, progress=False)
    px = raw["Close"].copy()
    px.columns = [c.replace(".NS", "") for c in px.columns]
    px = px.dropna(how="any")
    os.makedirs("data", exist_ok=True)
    px.to_csv(CACHE)
    return px


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def report(res, cfg, outdir="."):
    os.makedirs(f"{outdir}/figures", exist_ok=True)
    out = []
    out.append(f"Data: {res['train_range'][0].date()} to {res['test_range'][1].date()}  "
               f"| train ends {res['train_range'][1].date()} | test {res['test_range'][0].date()} to {res['test_range'][1].date()}")
    out.append(f"Pairs tested: {res['n_pairs_tested']}  | selected after FDR + half-life filter: {len(res['selected'])}\n")
    t = res["table"].copy()
    t = t[["a", "b", "pvalue", "p_adj", "beta", "half_life", "selected"]]
    for c in ("pvalue", "p_adj", "beta", "half_life"):
        t[c] = t[c].round(3)
    out.append("Top 10 pairs by Engle-Granger p-value (train only):\n" + md_table(t.head(10)) + "\n")
    if not res["oos_frames"]:
        out.append("No pairs passed selection. This is a valid result: report it honestly and "
                   "try a looser alpha or a different universe.")
        text = "\n".join(out); print(text)
        open(f"{outdir}/results.md", "w").write(text); return text

    is_net = portfolio_pnl(res["is_frames"], cfg.total_bps)
    oos_net = portfolio_pnl(res["oos_frames"], cfg.total_bps)
    oos_gross = portfolio_pnl(res["oos_frames"], 0.0)
    rows = []
    for name, p in (("In-sample (train), net", is_net), ("Out-of-sample (test), gross", oos_gross),
                    ("Out-of-sample (test), net", oos_net)):
        m = metrics(p, cfg.ann)
        rows.append(dict(Period=name, **{"Ann. return": f"{m['ann_return']:.1%}", "Ann. vol": f"{m['ann_vol']:.1%}",
                    "Sharpe": f"{m['sharpe']:.2f}", "Max DD": f"{m['max_drawdown']:.1%}"}))
    out.append(f"Portfolio metrics (costs: {cfg.total_bps:.0f} bps per side incl. slippage):\n" + md_table(pd.DataFrame(rows)) + "\n")

    pr = []
    for k, f in res["oos_frames"].items():
        n = f["gross"] - f["turnover"] * cfg.total_bps / 1e4
        pr.append(dict(Pair=k, Trades=n_trades(f), **{"Net Sharpe (test)": f"{metrics(n, cfg.ann)['sharpe']:.2f}",
                    "Net total return": f"{metrics(n, cfg.ann)['total_return']:.1%}"}))
    out.append("Per-pair out-of-sample results:\n" + md_table(pd.DataFrame(pr)) + "\n")

    srows = []
    for bps in (0, 5, 10, 20, 40):
        m = metrics(portfolio_pnl(res["oos_frames"], bps), cfg.ann)
        srows.append({"Cost per side (bps)": bps, "Ann. return": f"{m['ann_return']:.1%}", "Sharpe": f"{m['sharpe']:.2f}"})
    out.append("Out-of-sample sensitivity to transaction costs:\n" + md_table(pd.DataFrame(srows)))
    text = "\n".join(out); print(text)
    open(f"{outdir}/results.md", "w").write(text)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    (1 + is_net).cumprod().plot(ax=ax, label="In-sample (net)", color="gray")
    base = (1 + is_net).cumprod().iloc[-1]
    (base * (1 + oos_net).cumprod()).plot(ax=ax, label="Out-of-sample (net)", color="tab:blue")
    (base * (1 + oos_gross).cumprod()).plot(ax=ax, label="Out-of-sample (gross)", color="tab:blue", ls="--", alpha=0.6)
    ax.axvline(res["test_range"][0], color="k", ls=":"); ax.set_ylabel("Growth of 1 (equal-weight pairs)")
    ax.set_title("Backtested pairs portfolio: train vs test (simulated)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/figures/equity_curve.png", dpi=150)

    best = max(res["oos_frames"], key=lambda k: n_trades(res["oos_frames"][k]))
    f = res["oos_frames"][best]
    fig, ax = plt.subplots(figsize=(9, 4))
    f["z"].plot(ax=ax, lw=0.8, color="k")
    for lvl, c in ((cfg.z_entry, "r"), (-cfg.z_entry, "r"), (cfg.z_exit, "g"), (-cfg.z_exit, "g")):
        ax.axhline(lvl, color=c, ls="--", lw=0.7)
    ax.fill_between(f.index, -5, 5, where=f["pos"] != 0, color="tab:blue", alpha=0.12, label="In position")
    ax.set_ylim(-5, 5); ax.set_title(f"Spread z-score, {best} (test period)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/figures/zscore_example.png", dpi=150)
    return text


if __name__ == "__main__":
    import sys
    prices = load_prices(refresh="--refresh" in sys.argv)
    report(run_pipeline(prices, Config()), Config())
