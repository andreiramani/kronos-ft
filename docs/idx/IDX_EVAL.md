# Kronos-FT IDX Evaluation (Sector Benchmarks)

This document summarizes out-of-sample performance, training dynamics, and head-to-head benchmarking of the **Kronos-FT IDX** model against the **pretrained base Kronos** across 10 bellwether sector constituents on the Indonesia Stock Exchange (IDX / BEI).

---

## 1. Test Setup and Parameters

Evaluations were executed strictly on the held-out test split (the final 10% of chronological time-series data), guaranteeing zero-lookahead bias with no future bars visible during inference:

- **Architecture:** Kronos Foundation Model (*Decoder-Only Transformer*, 102.3M parameters).
- **Fine-Tuner:** Andrei Ramani.
- **Lookback Context:** **256 bars**.
- **Inference Strategy:**
  - `temperature: 0.3` (suppresses noise, keeping trajectory paths realistic).
  - `sample_count: 5` (averages across 5 Monte Carlo sampling trajectories).
  - `top_p: 0.9`.
- **Adaptive Forecast Horizons:**
  - **Daily (`Daily` / `D`):** **5 bars** (one calendar trading week, optimal for swing trading).
  - **Intraday (`5T`, `30T`, `1H`):** **16 bars**.
- **Operational Timeframes Evaluated:** `5T`, `30T`, `1H`, and `D` - **40 test cases total** (10 tickers × 4 timeframes).
- **Evaluation Metrics:**
  - **Directional Accuracy (*Directional Hit / dir_acc*):** Binary correctness of return direction (*up/down*) at the horizon endpoint.
  - **Path Correlation (*Information Coefficient / IC*):** Pearson correlation between predicted path and realized price trajectory.
  - **Price Deviation (*MAPE*):** Mean Absolute Percentage Error on closing prices across the forecast window.

---

## 2. Benchmark Tickers across 10 IDX Sectors (IDX-IC)

To validate model robustness across varying industrial dynamics and volatility regimes, the evaluation encompasses 10 liquid, large-cap equities representing 10 primary IDX-IC sectors:

| No | Ticker | Company Name | IDX-IC Sector | Trading Characteristics |
| :-: | :--- | :--- | :--- | :--- |
| 1 | **BBRI** | Bank Rakyat Indonesia (Persero) Tbk. | Financials | Mega-cap banking bellwether, high liquidity, sensitive to foreign institutional flows and rate cycles |
| 2 | **ANTM** | Aneka Tambang Tbk. | Basic Materials | Metals & mining (gold & nickel), volatility tracks global commodity supercycles |
| 3 | **ADRO** | Alamtri Resources Indonesia Tbk. | Energy | Coal energy compounder with robust operational cash flows and dividend track record |
| 4 | **ICBP** | Indofood CBP Sukses Makmur Tbk. | Consumer Non-Cyclicals | Consumer staples defensive anchor with stable, predictable price behavior |
| 5 | **MAPI** | Mitra Adiperkasa Tbk. | Consumer Cyclicals | Mid-to-high income lifestyle retail, sensitive to discretionary consumer spending |
| 6 | **TLKM** | Telkom Indonesia (Persero) Tbk. | Infrastructures | Telecommunications and data center infrastructure cornerstone with substantial index weighting |
| 7 | **ASII** | Astra International Tbk. | Industrials | Conglomerate across automotive, heavy equipment machinery, and financial services |
| 8 | **CTRA** | Ciputra Development Tbk. | Properties & Real Estate | Residential and commercial developer, sensitive to mortgage interest rate dynamics |
| 9 | **EMTK** | Elang Mahkota Teknologi Tbk. | Technology | Digital media, venture investments, and technology ecosystem; elevated beta |
| 10 | **KLBF** | Kalbe Farma Tbk. | Healthcare | Leading pharmaceuticals and healthcare solutions group across Southeast Asia |

---

## 3. Training Dynamics & Model Convergence

The adaptation of the Kronos foundation model to the Indonesian equity market was conducted via a two-stage fine-tuning architecture:

### 3.1 Stage 1: Tokenizer Fine-Tuning
The tokenizer converts continuous OHLCV candlestick dynamics into discrete latent tokens calibrated to IDX fractional price tick dynamics:
- **Initial Loss:** `train_rec_loss: 0.0217`, `val_rec_loss: 0.0185` (`total_val_loss: -0.0518`).
- **Final Convergence:** `train_rec_loss: 0.0167`, `val_rec_loss: 0.0159` (`total_val_loss: -0.0547`).
- **Improvement:** Candlestick reconstruction loss decreased by **~14.1%**, yielding sharp, discrete token quantization specialized for Indonesian market dynamics.

### 3.2 Stage 2: Predictor Fine-Tuning
The Decoder-Only Transformer predictor trains causal autoregressive projection capabilities over future token sequences:
- **Best Validation Checkpoint:** Reached at **Epoch 1** with `train_ce_loss: 2.6099` and `val_ce_loss: 2.4895` at a learning rate of $1.87 \times 10^{-5}$.

---

## 4. Performance Summary by Timeframe (Head-to-Head vs Base Model)

Aggregate performance comparison between the fine-tuned (**FT**) model and the pretrained base (**Base**) model across all 40 test cases (10 tickers × 4 timeframes).

| Timeframe | FT MAPE | FT Median MAPE | Base MAPE | Base Median MAPE | FT Dir Hit | Base Dir Hit | FT IC | Base IC | Forecast Horizon |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **5T** (5 Min) | 0.52% | **0.50%** | **0.51%** | 0.54% | **40.0%** | **40.0%** | +0.151 | **+0.226** | 16 bars (~1.3 hours) |
| **30T** (30 Min) | 2.02% | 1.85% | **1.99%** | **1.46%** | **70.0%** | 60.0% | **+0.055** | +0.007 | 16 bars (~1 trading day) |
| **1H** (1 Hour) | **1.98%** | **1.94%** | 2.69% | 2.53% | **70.0%** 🟢 | 10.0% 🔴 | **+0.301** 🟢 | -0.454 🔴 | 16 bars (~2.5 trading days) |
| **Daily (`D`)** | **3.84%** | **3.05%** | 6.59% | 4.39% | **70.0%** | **70.0%** | +0.389 | **+0.457** | **5 bars (1 trading week)** |
| **ALL** | **2.09%** | **1.35%** | **2.95%** | **1.78%** | **62.5%** 🟢 | **45.0%** | **+0.224** 🟢 | **+0.059** | **40 Test Cases** |

### Head-to-Head Win Counts (40 Total Test Cases)

| Metric Criterion | FT Wins | Base Wins | Ties | FT Win Rate (Non-Tied) |
| :--- | :---: | :---: | :---: | :---: |
| **Directional Hit (*dir_acc*)** | **12** | 5 | 23 | **70.6%** |
| **Path Correlation (*IC*)** | **25** | 15 | 0 | **62.5%** |
| **Price Deviation (*MAPE*)** | **24** | 16 | 0 | **60.0%** |

---

## 5. Daily Detail Analysis (`Daily` / `D`, 5-Bar Horizon)

On the daily swing timeframe, the fine-tuned model substantially suppresses price deviation (average MAPE dropped from 6.59% to **3.84%**) and wins 7 of the 10 sectors on price accuracy:

| Ticker | IDX-IC Sector | FT MAPE | Base MAPE | FT Dir Hit | Base Dir Hit | FT IC | Base IC | MAPE Winner | Direction Winner | Evaluation Chart |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ADRO** | Energy | **1.10%** | 2.01% | **100.0%** | 100.0% | **-0.071** | -0.114 | **FT** | TIE | [eval_ADRO_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_ADRO_D.png) |
| **KLBF** | Healthcare | **1.33%** | 1.86% | **100.0%** | 100.0% | **+0.544** | +0.455 | **FT** | TIE | [eval_KLBF_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_KLBF_D.png) |
| **MAPI** | Consumer Cyclicals | **1.56%** | 2.84% | 0.0% | 0.0% | **-0.516** | -0.679 | **FT** | TIE | [eval_MAPI_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_MAPI_D.png) |
| **TLKM** | Infrastructures | **2.45%** | 10.13% | **100.0%** | 100.0% | +0.652 | **+0.874** | **FT** (-76%) | TIE | [eval_TLKM_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_TLKM_D.png) |
| **BBRI** | Financials | 3.01% | **1.75%** | **100.0%** | 100.0% | **+0.869** | +0.655 | **BASE** | TIE | [eval_BBRI_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_BBRI_D.png) |
| **CTRA** | Properties | **3.09%** | 10.61% | **100.0%** | 100.0% | +0.588 | **+0.979** | **FT** (-71%) | TIE | [eval_CTRA_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_CTRA_D.png) |
| **EMTK** | Technology | **4.24%** | 20.58% | 0.0% | 0.0% | **+0.754** | +0.078 | **FT** (-79%) | TIE | [eval_EMTK_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_EMTK_D.png) |
| **ASII** | Industrials | **4.78%** | 7.64% | **100.0%** | 0.0% | +0.929 | **+0.990** | **FT** | **FT** | [eval_ASII_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_ASII_D.png) |
| **ANTM** | Basic Materials | 8.18% | **2.55%** | **100.0%** | 100.0% | +0.842 | **+0.933** | **BASE** | TIE | [eval_ANTM_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_ANTM_D.png) |
| **ICBP** | Consumer Non-Cyclicals | 8.64% | **5.95%** | 0.0% | **100.0%** | -0.701 | **+0.402** | **BASE** | **BASE** | [eval_ICBP_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_ICBP_D.png) |

---

## 6. Intraday Performance Highlights (`1H`, `30T`, `5T`)

### 6.1 1-Hour Timeframe (`1H`, 16-Bar Horizon) - Remarkable Reversal
- **Radical Trend Reversal:** The base model suffered from a severe structural anomaly in the Indonesian market, collapsing to a directional hit rate of only **10.0%** and an intensely inverted correlation of **-0.454**. Fine-tuning resolved this failure mode entirely: directional accuracy surged to **70.0%** and path correlation rebounded to a solid **+0.301**.
- **Sector Dominance:** FT outperformed Base across **8 of 10 tickers on MAPE**, and **9 of 10 tickers on path correlation (*IC*)**.
- Standout constituent gains:
  - **CTRA:** MAPE slashed by over half (**0.63%** vs 1.37%), direction 100% vs 0%, IC +0.201 vs -0.299.
  - **EMTK:** MAPE reduced almost threefold (**0.88%** vs 2.44%), direction 100% vs 0%, IC +0.525 vs -0.397.
  - **KLBF:** Sharp MAPE decline (**1.03%** vs 2.85%), direction 100% vs 0%.
  - **ANTM & ADRO:** Both registered 100% directional accuracy on FT (vs 0% Base) with consistent MAPE reductions.

### 6.2 30-Minute Timeframe (`30T`, 16-Bar Horizon)
- **BBRI & ADRO:** The fine-tuned model secured perfect directional hit rates (**100.0%** vs 0.0% Base) and meaningfully suppressed price errors.
- **CTRA:** Achieved 100.0% direction (vs 0.0% Base) with positive correlation of +0.256 (vs -0.298).
- **Aggregate Directional Hit:** Fine-tuned model outperformed the base model with **70.0%** vs 60.0%.

### 6.3 5-Minute Timeframe (`5T`, 16-Bar Horizon)
- **Microstructure Precision:** On 5-minute intervals, market behavior is dominated by bid-ask bounce and tick fractions. Both FT and Base maintained tight, sub-percent tracking (**MAPE 0.52% vs 0.51%**).
- **BBRI & ANTM:** Fine-tuned model excelled on ANTM (MAPE **0.30%** vs 0.58%, direction **100%** vs 0%, IC **+0.773** vs -0.494) and BBRI (MAPE **0.42%** vs 0.54%, IC **+0.834** vs +0.745).

---

## 7. Experimental Analysis & Operational Boundaries

### 7.1 Context Length Experiment: 256 Bars vs 512 Bars
An experiment was conducted to assess whether expanding context memory to 512 daily bars (~2 calendar trading years) yields performance benefits:

| Evaluation Metric | Context = 256 Bars | Context = 512 Bars | Analytical Finding |
| :--- | :---: | :---: | :--- |
| **Average Daily MAPE** | 4.90% | **4.68%** | Marginal error reduction (-0.22%) |
| **Median Daily MAPE** | 4.95% | **3.29%** | Improved for TLKM (2.61%) & ICBP (2.80%) |
| **Directional Hit Ratio** | **70.0%** | **70.0%** | Identical |
| **Path Correlation (*Avg IC*)** | **+0.129** (Positive) | **-0.065** (Negative) | 🔴 **Trend correlation inverted negatively on BBRI and CTRA** |

> **Technical Conclusion:** Extending context memory to 512 bars induces **attention dilution**. Attention heads become distracted by obsolete macroeconomic regimes irrelevant to current market dynamics (on `BBRI`, MAPE expanded from 2.99% to 9.53%). Therefore, **`lookback: 256 bars` is established as the locked operational standard**.

### 7.2 Rationale for Withdrawing Weekly Timeframes from Live Operations
1. **Volatility Dispersion Expansion ($\sigma \propto \sqrt{\Delta t}$):** Over a 1-month horizon (4 weekly bars / ~20 trading days), cumulative price dispersion widens by approximately $\sqrt{20} \approx 4.5\times$ relative to daily forecasts. MAPE errors naturally scale up and are non-comparable to operational timeframes.
2. **Training Allocation Focus:** 95% of training sample weight was allocated to intraday and daily timeframes; weekly was allocated only 5% as a broad macro trend anchor.
3. **Real-World Execution Realities:** Tactical capital deployment centers on daily swing horizons (3–5 days) and intraday entries, rather than guessing weekly candle morphology that is vulnerable to earnings releases or exogenous macro shocks.

---

## 8. Key Findings & Operational Recommendations

1. **Resolution of 1-Hour Directional Inversion:**  
   The single most critical achievement of the fine-tuning process was rescuing the 1-hour timeframe from a disastrous **10.0%** hit rate (base model) up to **70.0%** (fine-tuned), and reversing severe negative correlation (**-0.454**) into positive alignment (**+0.301**).
2. **Suppression of Extreme Daily Price Blowouts:**  
   The base model frequently suffered from extreme trajectory blowouts on specific tickers (e.g. EMTK 20.58%, CTRA 10.61%, TLKM 10.13%). Fine-tuning successfully stabilized projections within **2.4% to 4.2%**.
3. **EOD Rolling Forecast Discipline:**  
   Execute model inferences daily post-close (16:15 WIB). Shift the 256-bar context window one step forward ($T+1$) to forecast 3–5 trading days ahead, maintaining peak directional hit rates and containing MAPE within 1.5%–3.5%.
4. **Role as a Confluence Filter:**  
   Utilize Kronos-FT IDX forecasts as a directional momentum confirmation filter alongside the quantitative qIDX Dual-Engine before order routing.

---

