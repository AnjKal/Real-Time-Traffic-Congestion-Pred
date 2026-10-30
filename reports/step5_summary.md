# Step 5 - test-set evaluation

Test windows: 8,637. Units: km/h (MAE).

| Model | 5-min MAE | 10-min MAE | 20-min MAE |
|---|---|---|---|
| DCRNN | 2.14 | 2.66 | 3.01 |
| Persistence | 2.28 | 3.04 | 3.72 |
| Hist. Average | 3.93 | 3.93 | 3.93 |
| Last-Week | 4.29 | 4.29 | 4.29 |

**Verdict:** DCRNN beats all baselines at all 3 horizons -> PASS

DCRNN overall: MAE 2.52, RMSE 3.30, MAPE 15.8%
Rush MAE 2.44 vs off-peak 2.55; weekday 2.53 vs weekend 2.49
