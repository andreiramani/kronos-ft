# Kronos-FT TSE Production Playbook (Institutional Mock Live Trading Implementation)

This guide outlines architectural best practices, execution protocols, and operational risk controls for deploying the **Kronos-FT TSE** quantitative foundation model into production mock trading workflows across the Tokyo Stock Exchange (TSE / Kabutocho, Japan).

---

## 1. Model Role in the Production Architecture

Kronos-FT TSE is an autoregressive time-series foundation model designed to model non-stationary financial candlestick dynamics. Within an institutional quantitative trading framework, the model does not operate as an autonomous unconstrained black-box executor. Instead, it functions as a **directional confirmation filter and forward momentum validator**:

- **Core Operational Focus:** Supports multi-day **Daily Swing Trading (`Daily` / 3–5 trading day horizon)** and tactical **intraday execution timing (`5T`, `15T`, `30T`, `1H` / 16 bars)**.
- **Signal Confluence:** Projected trajectories validate whether multi-day trend momentum aligns with primary quantitative screening algorithms (e.g., trend breakout, volatility contraction, institutional order block retests, moving average confluence) prior to dispatching orders to broker APIs.
- **Ensemble Foundation:** Utilizes the promoted **$\alpha=0.50$** checkpoint (`models/tse_model.safetensors`), which combines local Japanese market price action adaptation with pretrained foundation model stability across multi-horizon cycles.

---

## 2. Locked Production Specifications

Based on extensive empirical evaluations across 180 multi-window out-of-sample benchmark scenarios, the following production inference parameters are locked to maintain elevated directional accuracy while minimizing forecast error (achieving a **51.0% relative return error reduction** vs Base):

| Parameter | Production Value | Architectural Justification |
| :--- | :---: | :--- |
| **`lookback_window`** | **256 bars** | Optimal context length; preserves positive path correlation (IC) without inducing attention dilution from stale macro regimes. |
| **`pred_len` (Daily Swing)** | **5 days** | Exactly one full trading week; captures a complete institutional swing cycle and prevents compounding trajectory drift. |
| **`pred_len` (Weekly Context)** | **4 bars** | One full trading month; used for secondary macro trend gating. |
| **`pred_len` (Intraday Timing)** | **16 bars** | Captures approx. 1.5 to 2.5 full TSE trading sessions (e.g., 16 hours or 16 thirty-minute bars). |
| **`temperature_daily`** | **0.30** | Calibrated sampling temperature for daily swing forecasting; prevents trajectory flatlining while avoiding excessive dispersion. |
| **`temperature_intraday`**| **0.20** | Low-entropy sampling for intraday timeframes (5T, 15T, 30T, 1H); suppresses high-frequency microstructure noise and autoregressive drift. |
| **`top_p` (Nucleus)** | **0.85 – 0.90** | Eliminates improbable token tails while retaining legitimate volatility expansion spikes. |
| **`sample_count`** | **5 – 10** | Averages across 5 (intraday) or 10 (swing) Monte Carlo projection paths to stabilize price level estimates against single-sample variance. |

---

## 3. Operational Discipline: EOD Rolling Forecast Protocol

The foundational discipline for deploying autoregressive foundation models in live environments: **never allow forecasts to become stale**.

```
┌────────────────────────────────────────────────────────┐
│ 1. POST-MARKET DATA INGESTION (16:00 JST)              │
│ - Collect official TSE closing prints (09:00-15:30)    │
│ - Validate data integrity (zero NaNs, volume > 0)      │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. ROLLING CONTEXT WINDOW SLIDE (T+1)                  │
│ - Extract trailing 256 daily bars                      │
│ - Ingest into KronosPredictor on GPU (cuda:0)          │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. 5-DAY FORWARD SWING INFERENCE                       │
│ - Execute Monte Carlo forward pass (T=0.30, S=10)      │
│ - Extract expected Close[t+5], Direction, and IC       │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. MULTI-ENGINE CONFLUENCE FILTER                      │
│ - Execute Long ONLY IF:                                │
│   (Kronos Direction == UP) AND (Quant Alpha >= 70)     │
└────────────────────────────────────────────────────────┘
```

1. **Daily Execution Schedule (16:00 JST / Post-Market Close):**
   - As soon as the Tokyo Stock Exchange closing auction (*Itayose*) concludes at 15:30 JST and official closing prices are published by 16:00 JST, ingest official daily market data (`open`, `high`, `low`, `close`, `volume`, `amount`) into the database.
2. **Extract the Trailing 256 Bars:**
   - Retrieve the most recent 256 bars for candidate tickers, then execute `KronosPredictor.from_pretrained(models_dir, exchange="tse")` to forecast the upcoming **5 trading days**.
3. **Order Confluence Rule:**
   - A buy order may only be placed when both conditions are satisfied concurrently:
     $$\text{Kronos Directional Forecast } (P_{t+5} > P_t) \quad \text{AND} \quad \text{Quantitative Confluence Score} \ge 70$$
4. **Shift Context Window Daily (Rolling Updates):**
   - After each trading session concludes, shift the 256-bar memory window forward by one step ($T+1$). This ensures all inferences are conditioned on the freshest market prints, keeping realized daily return error bounded around **3.5% to 5.2%**.

---

## 4. Alignment with TSE Market Microstructure Regulations

AI foundation models process quantitative time-series patterns and do not natively account for exchange-specific microstructure constraints. The execution engine must strictly enforce Tokyo Stock Exchange trading regulations:

### 4.1. Official TSE Price Tick Table (Yen Fractional Ticks)
The Tokyo Stock Exchange enforces tiered price tick increments. Orders sent to exchange gateways must be rounded according to official TSE tick tables:

* **TOPIX 100 Constituents (Large Cap / Decimal Ticks):**
  * Price $\le ¥1,000$: Tick is **¥0.1**
  * $¥1,000 < \text{Price} \le ¥3,000$: Tick is **¥0.5**
  * $¥3,000 < \text{Price} \le ¥10,000$: Tick is **¥1.0**
  * $¥10,000 < \text{Price} \le ¥30,000$: Tick is **¥5.0**
  * $¥30,000 < \text{Price} \le ¥100,000$: Tick is **¥10.0**
* **Other TSE Prime Constituents (Standard Ticks):**
  * Price $\le ¥3,000$: Tick is **¥1.0**
  * $¥3,000 < \text{Price} \le ¥5,000$: Tick is **¥5.0**
  * $¥5,000 < \text{Price} \le ¥30,000$: Tick is **¥10.0**
* **Rounding Discipline:**
  * Take Profit (TP) limit orders must be rounded down to the nearest valid tick (*floor tick*).
  * Stop Loss (SL) triggers must be rounded down (*floor tick*).

### 4.2. Session Boundary Handling (Zenba / Goba / Lunch Recess)
* **Morning Session (*Zenba*):** 09:00 – 11:30 JST.
* **Lunch Recess:** 11:30 – 12:30 JST (No trading occurs; zero volume). Do not insert synthetic bars across this gap.
* **Afternoon Session (*Goba*):** 12:30 – 15:30 JST (Note: TSE extended closing time from 15:00 to 15:30 JST in November 2024).
* Intraday execution bars (e.g. 5T, 15T, 30T) must be mapped to chronological active trading bars only.

### 4.3. Anti-Chasing & Morning Gap Limits
* **Maximum Opening Gap Limit:** Do not chase market buy orders if the opening print gaps up beyond:
  $$P_{\text{open}} > P_{\text{close}} + 0.5 \times \text{ATR}_{14}$$
  Wait for a pullback to VWAP during the 09:30–10:30 JST morning momentum window.

### 4.4. Corporate Actions & Exclusion Filters
* **Special Quotation (*Tokuhai* / Special Quotes):** When a stock enters *Tokubetsu Kehai* (matching halt due to order imbalance), suspend automated execution until continuous auction (*Zaraba*) resumes.
* **Securities Under Supervision / Delisting Post:** Stocks placed on the TSE Supervisory Post (*Kanri Meigara*) or Liquidation Post (*Seiri Meigara*) **must be 100% blocked**.
* **Dividend Blackout Window:** Avoid opening swing positions within $[\text{ExDate}-2\text{d}, \text{ExDate}+1\text{d}]$ for high-yield equities ($\ge 3.0\%$ yield) to prevent post-dividend price drop drawdowns.

---

## 5. Usage Boundaries & High-Beta Outlier Management

> [!CAUTION]
> **Operational Risk Warning: SoftBank Group (9984.T) and High-Beta Tech Holdings**  
> While Kronos-FT TSE v2 successfully stabilized the SoftBank Group anomaly (slashing MAPE from 23.53% to 5.40%), tickers with significant global tech venture holdings (Arm, Vision Fund, private AI investments) are heavily influenced by overnight US tech fluctuations (NASDAQ) and USD/JPY foreign exchange swings rather than domestic Japanese cash equity dynamics.
> 
> **Execution Rule:** For institutional swing trading, allocate primary capital to domestic bellwethers with high model precision (e.g., Toyota Motor 7203.T, Mitsubishi UFJ 8306.T, Tokyo Electron 8035.T, Shin-Etsu 4063.T, and the Nikkei 225 ETF 1321.T). Apply 50% reduced position sizing when executing signals on 9984.T.

---

## Disclaimer

> [!WARNING]
> **NOT FINANCIAL OR INVESTMENT ADVICE:**  
> All model outputs, price trajectories, candlestick forecasts, quantitative metrics, and evaluations produced by **Kronos-FT** are provided strictly for academic research, artificial intelligence experimentation, and financial time-series modeling studies. **None of the data, projections, or insights generated constitute formal financial advice, investment recommendations, endorsements, or legal solicitation to buy or sell securities.**  
>
> Trading financial instruments across equities markets (including IDX, NASDAQ, TSE, and NSE) involves substantial volatility and high risk of capital loss. Historical backtesting and past performance do not guarantee future results or profitability.  
>
> Users bear sole and full responsibility for all trading decisions, independent analysis (*Do Your Own Research* / DYOR), and risk management execution. The developers and contributors assume no liability whatsoever for any direct or indirect financial losses, damages, or costs incurred as a result of using this software or these models.

