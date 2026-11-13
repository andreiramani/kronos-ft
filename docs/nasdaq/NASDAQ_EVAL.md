# Kronos-FT NASDAQ Evaluation (10 Institutional Benchmarks)

This document summarizes the out-of-sample performance of the **Kronos-FT NASDAQ** model (*Stage 2 Predictor*) against the **pretrained base Kronos** on 10 large-cap NASDAQ benchmarks.

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
| 1 | **AAPL** | Apple Inc. | Mega-cap Tech | Hardware/services bellwether, low beta relative to peers |
| 2 | **NVDA** | NVIDIA Corp. | Semiconductors / AI | Highest-beta AI accelerator leader |
| 3 | **MSFT** | Microsoft Corp. | Software / Cloud | Azure + enterprise software compounder |
| 4 | **AMZN** | Amazon.com Inc. | E-commerce / Cloud | Retail margin cycle plus AWS |
| 5 | **GOOGL** | Alphabet Inc. | Internet / Advertising | Search and cloud advertising cyclical |
| 6 | **META** | Meta Platforms Inc. | Social Media | Ad-spend beta, high capex sensitivity |
| 7 | **TSLA** | Tesla Inc. | EV / Auto | Retail-flow driven, widest dispersion |
| 8 | **AMD** | Advanced Micro Devices | Semiconductors | AI/datacenter challenger, high volatility |
| 9 | **COST** | Costco Wholesale | Consumer Staples | Defensive, low-volatility retail |
| 10 | **QQQ** | Invesco QQQ Trust | Index ETF | NASDAQ-100 benchmark anchor |

---

## 3. Performance Summary by Timeframe

Averages over the 10 tickers per timeframe. FT = fine-tuned, Base = pretrained Kronos.

| Timeframe | FT MAPE | Base MAPE | FT Dir Hit | Base Dir Hit | FT IC | Base IC | Horizon |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **5T** (5 Min) | 0.41% | **0.21%** | 30.0% | **60.0%** | -0.155 | **+0.220** | 16 bars |
| **30T** (30 Min) | 1.86% | **1.44%** | 60.0% | **70.0%** | -0.071 | **+0.074** | 16 bars |
| **1H** (1 Hour) | 1.95% | **1.58%** | 60.0% | **70.0%** | **+0.121** | +0.392 | 16 bars |
| **D** (Daily) | **5.36%** | 6.43% | **50.0%** | 30.0% | **-0.255** | -0.122 | 5 bars |
| **W** (Weekly) | **8.96%** | 18.07% | 40.0% | **60.0%** | -0.031 | **+0.069** | 4 bars |
| **ALL** | **3.71%** | 5.54% | 48.0% | **58.0%** | -0.078 | **+0.127** | 50 cases |

**Win counts (50 cases):**

| Criterion | FT wins | Base wins | Ties |
| :--- | :---: | :---: | :---: |
| Directional hit | 15 | 20 | 15 |
| Correlation (IC) | 18 | 32 | 0 |
| MAPE | 20 | 30 | 0 |

Reading: fine-tuning halves weekly MAPE but **does not** improve short-horizon intraday accuracy on NASDAQ. The base model remains stronger on `5T`–`1H` direction and IC.

---

## 4. Daily Detail (`D`, 5-bar horizon)

| Ticker | FT MAPE | Base MAPE | FT Dir | Base Dir | FT IC | Base IC | MAPE Winner |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AAPL** | **1.32%** | 8.55% | 0% | 0% | -0.863 | -0.692 | FT |
| **COST** | **2.06%** | 5.50% | 0% | 0% | -0.105 | -0.750 | FT |
| **MSFT** | 2.49% | **2.37%** | 100% | 100% | 0.151 | 0.344 | Base |
| **META** | 2.94% | **2.71%** | **100%** | 0% | -0.823 | 0.781 | Base |
| **QQQ** | **3.00%** | 3.67% | **100%** | 0% | -0.949 | -0.737 | FT |
| **AMZN** | 3.84% | **2.37%** | 0% | 0% | -0.777 | -0.664 | Base |
| **TSLA** | **5.16%** | 9.61% | 100% | 100% | 0.766 | 0.985 | FT |
| **GOOGL** | 7.35% | **2.84%** | 100% | 100% | 0.917 | 0.913 | Base |
| **AMD** | 12.39% | **10.92%** | 0% | 0% | -0.005 | -0.693 | Base |
| **NVDA** | **13.04%** | 15.74% | 0% | 0% | -0.867 | -0.702 | FT |

Per-ticker MAPE is bimodal: FT wins decisively on AAPL, COST, TSLA, NVDA, QQQ; Base wins on GOOGL, AMZN, AMD, MSFT, META. Only 5 of 10 daily cases show a positive FT IC.

---

## 5. Analysis

### Where fine-tuning pays off
- **Weekly MAPE:** FT 8.96% vs Base 18.07%, with FT winning 7 of 10 weekly cases. This is the only timeframe with a margin large enough (9.11pp) to offset the intraday losses, where Base wins 22 of 30. It is **not** evidence of an overall MAPE advantage: across all 50 cases FT wins 20 and Base wins 30.
- **Daily MAPE:** FT 5.36% vs Base 6.43%, driven by AAPL (1.32% vs 8.55%) and TSLA (5.16% vs 9.61%).
- **Low-volatility names:** COST and AAPL compress cleanly; the fine-tuned tokenizer is well calibrated for narrow-range series.

### Where fine-tuning does not help
- **Intraday direction and IC:** Base is stronger at `5T` and `30T` on both IC and hit rate. The intraday test windows are short (16 bars) and the sample is a single endpoint, so direction is effectively a coin flip; treat these numbers as noise-dominated.
- **High-beta AI names:** NVDA and AMD remain the worst cases at daily horizon (MAPE 13.04% and 12.39%), despite modest FT gains over Base. Path volatility dominates.
- **Negative IC with correct direction:** QQQ and META show 100% directional hit with negative IC — the endpoint was right but the intermediate path was systematically wrong. Do not read direction alone as path quality.

### Caveat on sample size
Each cell is one ticker, one window, one prediction. Directional hit is a binary outcome, so its standard error is roughly ±16pp at n=10. Averages across 50 cases are the only figures worth quoting.

---

## 6. Files and Evidence

- **Full CSV summary:**  
  [`outputs/evaluations/nasdaq/evaluation_summary.csv`](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/nasdaq/evaluation_summary.csv)
- **Charts (50 PNG, actual vs FT vs Base):**  
  [`outputs/evaluations/nasdaq/`](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/nasdaq/), naming pattern `eval_{TICKER}_{TIMEFRAME}.png`.
- **Config:** [`configs/nasdaq/config_nasdaq.yaml`](file:///E:/anaconda/AR/kronos-ft/configs/nasdaq/config_nasdaq.yaml)
- **Evaluation script:** [`scripts/nasdaq/04_eval_nasdaq.py`](file:///E:/anaconda/AR/kronos-ft/scripts/nasdaq/04_eval_nasdaq.py)
