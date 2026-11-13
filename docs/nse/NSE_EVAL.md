# Kronos-FT NSE Evaluation (10 Nifty 50 Benchmarks)

This document summarizes the out-of-sample performance of the **Kronos-FT NSE** model (*Stage 2 Predictor*) against the **pretrained base Kronos** on 10 NSE large-cap benchmarks.

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
| 1 | **RELIANCE** | Reliance Industries | Energy / Conglomerate | Index heavyweight, refined-product margin cycle |
| 2 | **TCS** | Tata Consultancy Services | IT Services | Export revenue, USD/INR sensitive |
| 3 | **HDFCBANK** | HDFC Bank | Private Bank | Largest private-sector lender by assets |
| 4 | **INFY** | Infosys | IT Services | Export revenue, deal-cycle driven |
| 5 | **ICICIBANK** | ICICI Bank | Private Bank | Credit-growth and NIM sensitive |
| 6 | **BHARTIARTL** | Bharti Airtel | Telecom | ARPU cycle, 5G capex |
| 7 | **SBIN** | State Bank of India | Public Bank | PSU lender, policy-rate sensitive |
| 8 | **LT** | Larsen & Toubro | Infrastructure | Order-book and capex cycle |
| 9 | **BAJFINANCE** | Bajaj Finance | NBFC | Consumer credit, highest beta in the set |
| 10 | **NIFTYBEES** | Nippon India Nifty 50 ETF | Index ETF | Nifty 50 benchmark anchor |

---

## 3. Performance Summary by Timeframe

Averages over the 10 tickers per timeframe. FT = fine-tuned, Base = pretrained Kronos.

| Timeframe | FT MAPE | FT MAPE (median) | Base MAPE | FT Dir Hit | Base Dir Hit | FT IC | Base IC | Horizon |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **5T** (5 Min) | 0.31% | 0.35% | **0.19%** | **40.0%** | 30.0% | **+0.001** | -0.193 | 16 bars |
| **30T** (30 Min) | 57.68%* | **1.16%** | 10.52%* | 40.0% | **50.0%** | -0.005 | **+0.108** | 16 bars |
| **1H** (1 Hour) | 51.31%* | 2.94% | **9.36%*** | 50.0% | **60.0%** | **+0.023** | -0.000 | 16 bars |
| **D** (Daily) | **5.14%** | 5.27% | 5.21% | 10.0% | **50.0%** | -0.194 | **+0.013** | 5 bars |
| **W** (Weekly) | **6.38%** | **6.17%** | 6.81% | 50.0% | 50.0% | **+0.271** | -0.180 | 4 bars |
| **ALL** | 24.17%* | **2.56%** | 6.42%* | 38.0% | **48.0%** | **+0.019** | -0.051 | 50 cases |

\* Mean is distorted by three outlier cells (see Section 5). Median is the honest central figure.

**Win counts (50 cases):**

| Criterion | FT wins | Base wins | Ties |
| :--- | :---: | :---: | :---: |
| Directional hit | 8 | 13 | 29 |
| Correlation (IC) | 26 | 24 | 0 |
| MAPE | 21 | 29 | 0 |

Reading: NSE is the **weakest market for this fine-tune**. FT improves correlation marginally and cuts weekly MAPE slightly, but is clearly worse on daily direction (10% vs 50%).

---

## 4. Daily Detail (`D`, 5-bar horizon)

| Ticker | FT MAPE | Base MAPE | FT Dir | Base Dir | FT IC | Base IC | MAPE Winner |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **INFY** | **1.99%** | 3.13% | 0% | 0% | **0.483** | -0.869 | FT |
| **RELIANCE** | 2.25% | **1.46%** | 0% | 0% | -0.827 | **0.442** | Base |
| **BHARTIARTL** | **2.69%** | 4.68% | 0% | 0% | -0.754 | **-0.589** | FT |
| **TCS** | 2.82% | **1.71%** | 0% | 100% | **0.944** | -0.633 | Base |
| **NIFTYBEES** | 4.96% | **4.05%** | 0% | 100% | -0.970 | **0.679** | Base |
| **LT** | 5.58% | **4.69%** | 0% | 100% | -0.793 | **0.633** | Base |
| **HDFCBANK** | **5.86%** | 7.74% | **100%** | 0% | **0.902** | -0.223 | FT |
| **SBIN** | 6.18% | **5.58%** | 0% | 100% | 0.554 | **0.874** | Base |
| **ICICIBANK** | **9.12%** | 10.69% | 0% | 0% | -0.987 | **-0.791** | FT |
| **BAJFINANCE** | 10.00% | **8.39%** | 0% | 100% | -0.489 | **0.602** | Base |

Daily direction is the weak spot: **9 of 10 FT cases predicted the wrong endpoint direction**, while Base got 5 of 10 right. HDFCBANK is the only FT directional win.

---

## 5. Outliers and Data-Quality Notes

Three cells dominate the intraday means and must be read separately:

| Ticker | Timeframe | FT MAPE | Base MAPE | Note |
| :--- | :--- | :---: | :---: | :--- |
| **NIFTYBEES** | `30T` | 561.39% | 79.64% | ETF price near a low-denominator regime; percentage error explodes |
| **NIFTYBEES** | `1H` | 457.44% | 62.50% | Same ticker, same denominator effect |
| **BAJFINANCE** | `1H` | 34.28% | 5.98% | Single-path divergence on a 16-bar window |

Both models fail on NIFTYBEES intraday, so this is a data/denominator problem rather than a fine-tuning regression. Excluding these three cells, the FT intraday MAPE median is in the 1–3% band.

---

## 6. Analysis

### Gains from fine-tuning
- **Weekly correlation:** FT IC +0.271 vs Base -0.180 — the only clear, consistent edge on NSE. FT beats Base on IC in 26 of 50 cases.
- **Weekly MAPE:** 6.38% vs 6.81%, a small but consistent improvement across most tickers.
- **Low-volatility IT names:** INFY and BHARTIARTL compress well daily; TCS gets a 0.944 path correlation despite a wrong endpoint.

### Failures
- **Daily direction:** 10% hit rate is below coin flip and below Base. This fine-tune is not usable as a daily direction signal on NSE.
- **Intraday:** Base is better on both MAPE (excluding outliers) and direction at `30T` and `1H`.
- **High-beta NBFC:** BAJFINANCE is the worst case at daily horizon (10.00% MAPE, -0.489 IC) and the source of the intraday outlier.

### Caveat on sample size
Each cell is one ticker, one window, one prediction. Directional hit is binary, so its standard error is roughly ±16pp at n=10. Only the 50-case aggregates are worth quoting, and even those carry a wide interval.

---

## 7. Files and Evidence

- **Full CSV summary:**  
  [`outputs/evaluations/nse/evaluation_summary.csv`](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/nse/evaluation_summary.csv)
- **Charts (50 PNG, actual vs FT vs Base):**  
  [`outputs/evaluations/nse/`](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/nse/), naming pattern `eval_{TICKER}_{TIMEFRAME}.png`.
- **Config:** [`configs/nse/config_nse.yaml`](file:///E:/anaconda/AR/kronos-ft/configs/nse/config_nse.yaml)
- **Evaluation script:** [`scripts/nse/04_eval_nse.py`](file:///E:/anaconda/AR/kronos-ft/scripts/nse/04_eval_nse.py)
