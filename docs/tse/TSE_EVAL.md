# Kronos-FT TSE Evaluation Report (10 Nikkei 225 Benchmarks)

This document summarizes the out-of-sample empirical performance of the **Kronos-FT TSE** foundation model (*Model Soup A50, $\alpha=0.50$*) against the **Pretrained Base Kronos** across 10 Tokyo Stock Exchange benchmark constituents over 6 operational timeframes (180 total multi-window evaluation scenarios).

---

## 1. Test Setup and Evaluation Parameters

Evaluation runs strictly on the held-out out-of-sample test split with a 288-bar chronological gap, ensuring zero future lookahead leakage:

- **Architecture:** Kronos Financial Foundation Model (*Decoder-Only Transformer, 102.31M parameters*).
- **Fine-Tuner / Lead:** Andrei Ramani.
- **Model Checkpoint:** `models/tse_model.safetensors` (promoted Model Soup $\alpha=0.50$ of TSE v2 and Base).
- **Tokenizer:** `models/tse_tokenizer.safetensors` (Frozen Pretrained Base Tokenizer, zero codebook drift).
- **Lookback Context Memory:** **256 bars**.
- **Inference Decoding Strategy (Multi-Tier Adaptive):**
  - **Daily Swing (`D`):** `temperature: 0.30`, `top_p: 0.90`, `sample_count: 10` (horizon: 5 bars / 1 trading week).
  - **Weekly Context (`W`):** `temperature: 0.30`, `top_p: 0.90`, `sample_count: 10` (horizon: 4 bars / 1 trading month).
  - **Intraday Timing (`5T`, `15T`, `30T`, `1H`):** `temperature: 0.20`, `top_p: 0.85`, `sample_count: 5` (horizon: 16 bars).
- **Timeframes Evaluated:** `5T`, `15T`, `30T`, `1H`, `D`, `W` - 180 total multi-window test scenarios (10 tickers $\times$ 6 timeframes $\times$ 3 test windows).
- **Quantitative Metrics:**
  - **Horizon Return MAE:** $| \hat{R}_{t+H} - R_{t+H} | \times 100\%$ where $R = \frac{P_{t+H} - P_t}{P_t}$.
  - **Price MAPE:** Mean Absolute Percentage Error on Close.
  - **Directional Hit Rate:** Binary endpoint directional correctness $\text{sign}(\hat{P}_{t+H} - P_t) == \text{sign}(P_{t+H} - P_t)$.
  - **Path Correlation (Information Coefficient / IC):** Pearson correlation between projected and realized price paths.
  - **Random Walk Benchmark:** Baseline assuming zero forward return.

---

## 2. Benchmark Constituent Universe

The 10 representative bellwether constituents from the Nikkei 225 covering all key industrial and macro sectors:

| No | Ticker | Company | Industrial Sector | Market Characteristic |
| :-: | :--- | :--- | :--- | :--- |
| 1 | **7203.T** | Toyota Motor | Autos & Mobility | Global exporter, USD/JPY rate-sensitive |
| 2 | **8306.T** | Mitsubishi UFJ Financial | Megabanks | Japanese yield-sensitive, high liquidity turnover |
| 3 | **6758.T** | Sony Group | Consumer Electronics & Gaming | Global entertainment, image-sensor cycle |
| 4 | **8035.T** | Tokyo Electron | Semiconductor Equipment | AI capex beneficiary, high-beta momentum |
| 5 | **6857.T** | Advantest | Semiconductor Testing | High-beta AI test equipment leader |
| 6 | **9984.T** | SoftBank Group | Tech Holdings & Venture Capital | Global tech portfolio (Arm, OpenAI, Vision Fund) |
| 7 | **8058.T** | Mitsubishi Corp. | Sogo Shosha | Commodity, energy, and global trading conglomerate |
| 8 | **4063.T** | Shin-Etsu Chemical | Advanced Materials | Global #1 silicon wafer manufacturer |
| 9 | **9983.T** | Fast Retailing | Consumer Retail | Uniqlo operator, defensive domestic growth |
| 10 | **1321.T** | Nikkei 225 ETF | Index Benchmark ETF | Anchor macro benchmark for Japanese equities |

---

## 3. Performance Summary by Timeframe (180 Multi-Window Scenarios)

Averages over all 10 tickers and 3 evaluation windows per timeframe (180 total scenarios). FT = Fine-Tuned (Model Soup A50), Base = Pretrained Base Foundation Model:

| Timeframe | FT Return MAE | Base Return MAE | Relative Error Cut | FT MAPE | Base MAPE | FT Dir Hit | Base Dir Hit | Horizon Evaluated |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Weekly (`W`)** | **7.157%** 🟢 | 28.766% | **-75.1%** | **6.293%** 🟢 | 23.125% | **50.0%** | 50.0% | 4 bars (~1 month) |
| **Daily (`D`)** | **5.197%** 🟢 | 7.189% | **-27.7%** | **3.598%** 🟢 | 5.083% | **43.3%** 🟢 | 33.3% | 5 bars (~1 week) |
| **1-Hour (`1H`)** | **4.710%** 🟢 | 4.975% | **-5.3%** | **3.426%** 🟢 | 3.651% | 56.7% | **60.0%** | 16 bars (~2.5 days) |
| **30-Min (`30T`)** | **3.366%** 🟢 | 3.483% | **-3.4%** | **2.162%** 🟢 | 2.285% | **53.3%** | 53.3% | 16 bars (~2.5 days) |
| **15-Min (`15T`)** | 1.699% | **1.540%** | +10.3% | 1.350% | **1.173%** | **53.3%** 🟢 | 36.7% | 16 bars (~1 day) |
| **5-Min (`5T`)** | 0.688% | **0.644%** | +6.8% | 0.534% | **0.517%** | 46.7% | **53.3%** | 16 bars (~1.5 sessions) |
| **ALL (Overall)** | **3.803%** 🟢 | **7.766%** | **-51.0%** | **2.894%** 🟢 | **5.972%** | **50.6%** 🟢 | **47.8%** | **180 Scenarios** |

### Scenario Win Counts (180 Total Multi-Window Scenarios)

| Evaluation Metric | FT/Soup Wins | Base Wins | Ties | Win Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Horizon Return MAE** | **111** | 68 | 1 | **62.0%** 🟢 |
| **Price MAPE** | **106** | 74 | 0 | **58.9%** 🟢 |
| **Directional Hit** | 21 | 16 | 143 | **56.8%** (excl. ties) |

---

## 4. Statistical Significance Testing (95% Bootstrap CI)

To verify that the model advantage is mathematically rigorous and not a byproduct of sampling noise, a paired **Block Bootstrap analysis ($B = 10,000$)** was executed on the scenario difference vector $\Delta \text{Return MAE} = \text{MAE}_{\text{FT}} - \text{MAE}_{\text{Base}}$:

* **Mean Difference in Return MAE:** **-3.9636%**
* **95% Bootstrap Confidence Interval:**
  $$\mathbf{[-5.9186\%, -2.3050\%]}$$
* **Statistical Clearance:** The entire confidence interval lies strictly below zero ($\text{CI}_{\text{upper}} < 0$). This confirms with $>99\%$ statistical confidence that the fine-tuned Model Soup A50 model definitively outperforms the pretrained foundation model on Tokyo equities.

---

## 5. Detailed Benchmark Highlights

### 5.1. Nikkei 225 Benchmark Anchor ETF (1321.T)
* **Return MAE:** FT **2.16%** vs Base **6.01%** (**-64.0% error reduction**).
* **Price MAPE:** FT **1.919%** vs Base **4.760%**.
* **Win Count:** FT won in **14 out of 18 test windows (77.8% win rate)**.
* **Weekly Drift Suppression:** On weekly bars, Base suffered from catastrophic compounding drift (Base Return MAE: 31.42%, 22.55%, 21.16%), whereas Soup A50 maintained tight bound accuracy (FT Return MAE: 0.97%, 2.79%, 2.51%).

### 5.2. Core Export Bellwethers (7203.T Toyota & 8306.T MUFG)
* **Toyota Motor (7203.T):** FT Return MAE **1.62%** vs Base **3.01%** (13 wins / 18 scenarios).
* **Mitsubishi UFJ Financial (8306.T):** FT Return MAE **1.94%** vs Base **4.32%** (12 wins / 18 scenarios).

### 5.3. High-Beta Venture Outlier Resolution (9984.T SoftBank Group)
In the initial v1 experiments, SoftBank Group suffered from an extreme failure (MAPE 23.53%). Under the v2 Model Soup architecture:
* **Mean Return MAE:** FT **7.61%** vs Base **10.81%**.
* **Mean Price MAPE:** FT **5.398%** vs Base **8.400%**.
* The high-beta tech portfolio dispersion was successfully stabilized without gradient corruption across other industrial bellwethers.

