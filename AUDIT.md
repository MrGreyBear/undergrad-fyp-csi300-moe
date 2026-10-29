# Audit of the undergraduate FYP code (2026-10-28)

Scope: every script in `archive/undergrad_fyp/code/` checked against the thesis
`archive/undergrad_fyp/thesis/GROUP27JOJOyXIAOHUANGdeFYP.pdf`.

Conclusion: **no empirical number from the undergraduate thesis is carried into the new paper.**

## A. Look-ahead (forward) bias

| Script | Line | Issue | Affected thesis results |
|---|---|---|---|
| `5002xin.py` | 224 | `residual_correction = (test_close.values - moe_pred) * 0.6` adds 60% of the realised test error (incl. today's price) back into the forecast | Table 5.1 MoE RMSE 41.30 / DA 64.41%, Fig 5.1 |
| `5004.py` | 173 | same correction; DM test then compares LSTM with the corrected MoE | Table 5.3 DM = 2.94, p = 0.003 |
| `大数据集的最佳预测.py` | 216 | `cols = [target_col] + feature_cols`: future 3-day mean return used as an input feature; last input row overlaps 2 of the 3 target days | §5.7 rolling CV (~60% DA), Fig 5.12–5.13 |
| `数据导入2.py` | 257 | same | Fig 5.14, Ch. 6 style results |
| `5001加入印花税.py` | 204 | same | cost-adjusted variants |
| all rolling scripts | — | training targets near the fold boundary overlap the test window (no purge/embargo) | minor |

## B. Figures/tables not produced by a model

| Script | Line | Issue |
|---|---|---|
| `6001(5.2).py` | 285–288 | Fig 5.2 utilisation (0.92/0.05/0.03 vs 0.40/0.35/0.25) typed in ("Simulating data to demonstrate the concept") |
| `6001(5.2).py` | 307–312 | Fig 5.3 gating dynamics drawn with `linspace` |
| `数据导入.py` | whole file | synthetic gating stack plot from hand-set logits |
| `5004.py` | 220 | homogeneous-MoE RMSE computed by a formula from the other two RMSEs |
| `数据导入2.py` | 448–449 | Fig 5.14 XGBoost / Transformer per-fold returns typed in |

## C. Described in the thesis, absent from the code

ARIMA residual decomposition (MoE was fitted to scaled price levels), gate-entropy regularisation, uniform-prior mixing, concept bottleneck, decorrelation loss, α-fusion.

## D. No code found

Sharpe 2.86 backtest (Table 5.2), BTC test (Table 5.4), faithfulness (§5.3.1), stability (§5.3.2), Chapter 6 5-minute experiment and Wilcoxon test.

## E. Metric issues

- Directional accuracy computed as `sign(diff(pred)) == sign(diff(actual))` on price levels. For a lagging forecast this measures return autocorrelation, not forecasting skill.
- RMSE on price levels is dominated by the random-walk component; it cannot separate skill from persistence.
