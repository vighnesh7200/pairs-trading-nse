Data: 2018-01-01 to 2026-10-01  | train ends 2022-12-30 | test 2023-01-02 to 2026-10-01
Pairs tested: 45  | selected after FDR + half-life filter: 1

Top 10 pairs by Engle-Granger p-value (train only):
| a | b | pvalue | p_adj | beta | half_life | selected |
|---|---|---|---|---|---|---|
| HDFCBANK | KOTAKBANK | 0.002 | 0.085 | 0.918 | 21.522 | True |
| HDFCBANK | ICICIBANK | 0.054 | 0.77 | 0.466 | 34.406 | False |
| AXISBANK | FEDERALBNK | 0.06 | 0.77 | 0.667 | 43.826 | False |
| INDUSINDBK | PNB | 0.068 | 0.77 | 0.766 | 55.337 | False |
| AXISBANK | SBIN | 0.088 | 0.791 | 0.514 | 45.162 | False |
| KOTAKBANK | SBIN | 0.108 | 0.803 | 0.404 | 59.065 | False |
| AXISBANK | IDFCFIRSTB | 0.125 | 0.803 | 0.645 | 48.801 | False |
| HDFCBANK | SBIN | 0.169 | 0.824 | 0.436 | 62.513 | False |
| ICICIBANK | KOTAKBANK | 0.173 | 0.824 | 1.725 | 44.675 | False |
| PNB | SBIN | 0.183 | 0.824 | -0.171 | 123.91 | False |

Portfolio metrics (costs: 10 bps per side incl. slippage):
| Period | Ann. return | Ann. vol | Sharpe | Max DD |
|---|---|---|---|---|
| In-sample (train), net | 4.7% | 7.0% | 0.68 | -8.9% |
| Out-of-sample (test), gross | 8.5% | 7.4% | 1.15 | -6.0% |
| Out-of-sample (test), net | 7.4% | 7.4% | 1.00 | -6.2% |

Per-pair out-of-sample results:
| Pair | Trades | Net Sharpe (test) | Net total return |
|---|---|---|---|
| HDFCBANK/KOTAKBANK | 20 | 1.00 | 30.1% |

Out-of-sample sensitivity to transaction costs:
| Cost per side (bps) | Ann. return | Sharpe |
|---|---|---|
| 0 | 8.5% | 1.15 |
| 5 | 8.0% | 1.07 |
| 10 | 7.4% | 1.00 |
| 20 | 6.3% | 0.85 |
| 40 | 4.2% | 0.55 |