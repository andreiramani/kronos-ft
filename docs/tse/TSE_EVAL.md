# Kronos-FT TSE Evaluation (10 Nikkei 225 Benchmarks)

This document summarizes the out-of-sample performance of the **Kronos-FT TSE** model (*Stage 2 Predictor*) against the **pretrained base Kronos** on 10 Tokyo Stock Exchange benchmarks.

---

## 1. Test Setup and Parameters

Evaluation runs on the held-out test split (last 10% of each series, time-ordered), so no future bars are visible to the model at inference time:

- **Architecture:** Kronos Foundation Model (*decoder-only transformer*).
- **Fine-Tuner:** Andrei Ramani.
- **Lookback Context:** **256 bars**.
- **Inference Strategy:**
  - `temperature: 0.3`.
  - `sample_count: 5` (mean of 5 sampling paths).
  - `top_p: 0.9`.
- **Forecast Horizons:**
  - **Daily (`D`):** 5 bars (one trading week).
  - **Weekly (`W`):** 4 bars (one trading month).
  - **Intraday (`5T`, `30T`, `1H`):** 16 bars.
- **Timeframes Tested:** `5T`, `30T`, `1H`, `D`, `W` — 50 cases total (10 tickers × 5 timeframes).
- **Metrics:** Directional hit on the horizon endpoint, Pearson correlation (IC) of the predicted path, and MAPE on close.

---

## 2. Benchmark Tickers

| No | Ticker | Company | Segment | Characteristic |
| :-: | :--- | :--- | :--- | :--- |
| 1 | **7203.T** | Toyota Motor | Autos | Global exporter, JPY-sensitive |
| 2 | **8306.T** | Mitsubishi UFJ Financial | Megabank | Rate-sensitive, high turnover |
| 3 | **6758.T** | Sony Group | Consumer Electronics | Gaming and image-sensor cycle |
| 4 | **8035.T** | Tokyo Electron | Semiconductor Equipment | AI capex beneficiary |
| 5 | **6857.T** | Advantest | Semiconductor Testing | High-beta AI test demand |
| 6 | **9984.T** | SoftBank Group | Tech Holdings / VC | Portfolio-driven, widest dispersion |
| 7 | **8058.T** | Mitsubishi Corp. | Sogo Shosha | Commodity and trading conglomerate |
| 8 | **4063.T** | Shin-Etsu Chemical | Advanced Materials | Silicon wafer global leader |
| 9 | **9983.T** | Fast Retailing | Consumer Retail | Uniqlo operator, defensive growth |
| 10 | **1321.T** | Nikkei 225 ETF | Index ETF | Nikkei 225 benchmark anchor |

---

## 3. Performance Summary by Timeframe

Averages over the 10 tickers per timeframe. FT = fine-tuned, Base = pretrained Kronos.

| Timeframe | FT MAPE | FT MAPE (median) | Base MAPE | FT Dir Hit | Base Dir Hit | FT IC | Base IC | Horizon |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **5T** (5 Min) | **0.51%** | 0.39% | 0.57% | 50.0% | 50.0% | **+0.170** | -0.034 | 16 bars |
| **30T** (30 Min) | 1.45% | 1.09% | **1.38%** | **40.0%** | 10.0% | **-0.271** | -0.314 | 16 bars |
| **1H** (1 Hour) | 2.35% | 1.30% | **2.28%** | **70.0%** | 30.0% | -0.212 | **+0.240** | 16 bars |
| **D** (Daily) | **5.15%** | **2.76%** | 6.41% | **80.0%** | 40.0% | **+0.397** | -0.118 | 5 bars |
| **W** (Weekly) | **9.39%** | **7.42%** | 21.86% | 30.0% | **50.0%** | **+0.145** | +0.010 | 4 bars |
| **ALL** | **3.77%** | **1.68%** | 6.50% | **54.0%** | 36.0% | **+0.131** | -0.043 | 50 cases |

**Win counts (50 cases):**

| Criterion | FT wins | Base wins | Ties |
| :--- | :---: | :---: | :---: |
| Directional hit | 16 | 7 | 27 |
| Correlation (IC) | 29 | 21 | 0 |
| MAPE | 33 | 17 | 0 |

Reading: TSE is the **strongest market for this fine-tune**. FT beats Base on all three aggregate metrics and wins the MAPE comparison in 33 of 50 cases.

---

## 4. Daily Detail (`D`, 5-bar horizon)

| Ticker | FT MAPE | Base MAPE | FT Dir | Base Dir | FT IC | Base IC | MAPE Winner |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **7203.T** | **0.90%** | 8.43% | 100% | 100% | **0.667** | -0.532 | FT |
| **6758.T** | **1.13%** | 2.75% | **100%** | 0% | **-0.021** | -0.423 | FT |
| **9983.T** | **1.69%** | 3.91% | 100% | 100% | -0.812 | **-0.636** | FT |
| **8058.T** | **1.88%** | 3.49% | 0% | 0% | **0.925** | -0.911 | FT |
| **1321.T** | **2.62%** | 4.32% | **100%** | 0% | -0.883 | **0.724** | FT |
| **4063.T** | **2.89%** | 4.05% | 100% | 100% | **0.990** | 0.722 | FT |
| **8306.T** | 3.56% | **3.11%** | **100%** | 0% | **0.991** | -0.966 | Base |
| **8035.T** | **3.92%** | 7.15% | 100% | 100% | 0.935 | **0.979** | FT |
| **6857.T** | 9.44% | **5.17%** | **100%** | 0% | **0.527** | -0.798 | Base |
| **9984.T** | 23.53% | **21.70%** | 0% | 0% | 0.648 | **0.657** | Base |

FT wins 8 of 10 daily MAPE comparisons and hits direction in **8 of 10** cases (vs 4 of 10 for Base). 9984.T (SoftBank Group) is the only severe miss.

---

## 5. Analysis

### Gains from fine-tuning
- **Weekly MAPE:** FT 9.39% vs Base 21.86% — more than a halving, and FT wins 33 of 50 MAPE comparisons overall.
- **Daily direction:** 80% hit rate vs 40% for Base. 7203.T, 6758.T, 1321.T, 6857.T and 8306.T all flip from wrong to right.
- **Daily correlation:** FT IC +0.397 vs Base -0.118. 7203.T (0.667 vs -0.532), 8306.T (0.991 vs -0.966) and 8058.T (0.925 vs -0.911) show large path-quality reversals.
- **Low-volatility defensives:** 7203.T, 6758.T, 9983.T and 8058.T all land under 2% daily MAPE.

### Weak spots
- **Weekly direction:** 30% hit rate, below Base (50%). FT compresses weekly magnitude well but misreads the endpoint sign on the 4-bar horizon.
- **SoftBank Group (9984.T):** the only daily failure (23.53% MAPE, 0% direction). Portfolio-driven dispersion is not captured; treat this ticker as out of scope.
- **1H correlation:** Base is slightly better on IC at `1H` (+0.240 vs -0.212) even though FT wins on direction and is close on MAPE. Do not rely on FT path shape at `1H`.

### Caveat on sample size
Each cell is one ticker, one window, one prediction. Directional hit is binary, so its standard error is roughly ±16pp at n=10. The daily and weekly aggregates are the most stable; intraday direction figures are noise-dominated.

---

## 6. Files and Evidence

- **Full CSV summary:**  
  [`outputs/evaluations/tse/evaluation_summary.csv`](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/tse/evaluation_summary.csv)
- **Charts (50 PNG, actual vs FT vs Base):**  
  [`outputs/evaluations/tse/`](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/tse/), naming pattern `eval_{TICKER}_{TIMEFRAME}.png`.
- **Config:** [`configs/tse/config_tse.yaml`](file:///E:/anaconda/AR/kronos-ft/configs/tse/config_tse.yaml)
- **Evaluation script:** [`scripts/tse/04_eval_tse.py`](file:///E:/anaconda/AR/kronos-ft/scripts/tse/04_eval_tse.py)
