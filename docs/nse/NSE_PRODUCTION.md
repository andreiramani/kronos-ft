# Kronos-FT NSE Production Playbook (Institutional Mock Trading Implementation)

This guide outlines best practices, execution architecture, and operational risk controls for deploying the **Kronos-FT NSE** quantitative foundation model into mock trading workflows across the National Stock Exchange of India (NSE / Nifty 50).

---

## 1. Model Role in the Production Architecture

Kronos-FT NSE is an autoregressive time-series foundation model designed to process non-stationary financial candlestick dynamics. Within an institutional quantitative trading framework, the model functions as a **directional confirmation filter and forward momentum validator**, rather than an unconstrained autonomous black-box executor:

- **Core Operational Focus:** Supports multi-day **Daily Swing Trading (`Daily` / 3-5 trading day horizon)**, medium-term **Weekly Position Trading (`Weekly` / 4 bars)**, and tactical **intraday execution timing (`5T`, `30T`, `1H` / 16 bars)**.
- **Signal Confluence:** Projected trajectories validate whether multi-bar trend momentum aligns with primary quantitative screening rules (e.g., algorithmic momentum scoring, order book imbalance, volatility contraction, institutional liquidity sweeps) prior to dispatching orders to broker APIs.

---

## 2. Locked Production Specifications

Based on extensive empirical evaluations across 750 out-of-sample benchmark rolling windows, the following production inference parameters are locked to maintain elevated directional accuracy ($\ge 54.7\%$ Daily, $\ge 50.0\%$ 30T) while minimizing return forecast deviations (Return MAE $\le 3.04\%$ Daily, $\le 5.85\%$ Weekly):

| Parameter | Production Value | Architectural Justification |
| :--- | :---: | :--- |
| **`lookback_window`** | **256 bars** | Optimal context memory; preserves positive path correlation (IC) without inducing attention dilution from stale macro regimes. |
| **`pred_len` (Daily)** | **5 days** | Exactly one calendar trading week; represents a complete institutional swing cycle and prevents compounding trajectory drift. |
| **`pred_len` (Weekly)** | **4 weeks** | One calendar month; optimal horizon for multi-week position holding periods. |
| **`pred_len` (Intraday)** | **16 bars** | Captures balanced intraday windows (e.g., ~1.3 hours on 5T, 8 hours on 30T, ~2.5 sessions on 1H) for pullback timing. |
| **`temperature_daily`** | **0.30** | Calibrated sampling temperature for Daily and Weekly bars; models macro volatility without flatlining or excessive dispersion. |
| **`temperature_intraday`**| **0.15** | Low-entropy sampling; suppresses high-frequency microstructure noise for intraday execution bars (`5T`, `30T`, `1H`). |
| **`sample_count`** | **5** | Averages across 5 Monte Carlo projection paths to stabilize price level estimates against single-sample variance. |
| **`top_p`** | **0.90** | Standard nucleus sampling; eliminates improbable token tails while retaining legitimate volatility spikes. |

---

## 3. Operational Discipline: EOD Rolling Forecast

The foundational discipline for deploying autoregressive foundation models in production: **never allow forecasts to become stale**.

```
┌────────────────────────────────────────────────────────┐
│ 1. POST-MARKET DATA INGESTION (16:00 IST)              │
│ - Collect official NSE closing prints (09:15-15:30)    │
│ - Ingest official Bhavcopy & intraday OHLCV bars       │
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
│ - Execute Monte Carlo forward pass (T=0.30, S=5)       │
│ - Extract expected Close[t+5], Direction, and IC       │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. MULTI-ENGINE CONFLUENCE FILTER                      │
│ - Execute Long ONLY IF:                                │
│   (Kronos Direction == UP) AND (Quant Score >= 70)     │
└────────────────────────────────────────────────────────┘
```

### Detailed Daily Operational Steps

1. **Daily Execution Schedule (16:00 IST / Post-Market Close):**
   - The regular trading session on NSE concludes at 15:30 IST, and official volume-weighted average price (VWAP) closing prints are finalized by 15:45–16:00 IST.
   - Ingest official end-of-day OHLCV data into the database.
2. **Extract Trailing 256 Bars:**
   - Retrieve the most recent 256 bars for candidate tickers, then execute the inference engine to forecast the upcoming **5 trading days**.
3. **Order Confluence Rule:**
   - A buy order may only be executed when both conditions are satisfied concurrently:
     $$\text{Kronos Directional Forecast } (P_{t+5} > P_t) \quad \text{AND} \quad \text{Quantitative Momentum Score} \ge 70$$
4. **Shift Context Window Daily (Rolling Updates):**
   - After each trading day concludes, shift the 256-bar memory window forward by one step ($T+1$). This ensures all inferences are conditioned on the freshest market data, keeping realized daily return error tightly bounded between **1.8% and 3.0%**.

---

## 4. Alignment with NSE Market Microstructure Regulations

AI foundation models process quantitative time-series patterns and do not natively account for local market microstructure constraints. The execution engine must strictly enforce regulatory rules:

1. **Official NSE Tick Size Rounding (₹0.05):**
   - In accordance with NSE regulations, all Indian equities above ₹100 trade in discrete tick increments of **₹0.05** (or ₹0.01 for sub-₹100 stocks).
   - Take Profit (TP) targets generated from continuous float forecasts must always be rounded up (*ceil tick*).
   - Stop Loss (SL) triggers must always be rounded down (*floor tick*).

2. **Price Bands & Dynamic Circuit Limits:**
   - **Non-F&O Equities:** Subject to fixed daily operating bands (typically 5%, 10%, or 20%). Predicted target prices exceeding the daily circuit filter must be clamped to the upper band.
   - **F&O Equities:** Subject to dynamic price bands (initial 10% flexing band with cooling periods). Orders placed beyond the dynamic band will be rejected by the exchange matching engine.

3. **Anti-Chasing Opening Gap Filter:**
   - Pre-open order matching (09:00 - 09:08 IST) often results in significant opening gap-ups or gap-downs.
   - Maximum opening market buy price limit:
     $$P_{\text{limit}} \le P_{\text{close}} + 0.5 \times \text{ATR}_{14}$$
   - If a constituent gaps up by more than 2.0% at the 09:15 open, long execution must be paused to prevent buying at the exhaustion high of the opening auction.

4. **Surveillance & Corporate Actions Safeguards:**
   - **ASM / GSM Framework Filter:** Tickers placed under SEBI's **Additional Surveillance Measure (ASM)** or **Graded Surveillance Measure (GSM)** frameworks exhibit 100% margin requirements and trading restrictions. All ASM/GSM tickers **must be 100% excluded** from automated execution.
   - **Dividend Blackout Window:** Avoid establishing long positions during the $[\text{ExDate}-2\text{d}, \text{ExDate}+1\text{d}]$ window for equities with high dividend payout yields ($\ge 2.5\%$) to avoid mechanical ex-dividend price drops.
   - **Monthly F&O Expiry Volatility:** On the last Thursday of each month (or Wednesday if Thursday is a trading holiday), index and stock derivatives expire. Exercise caution and reduce intraday position sizing due to rollover-induced price swings.

---

## 5. Usage Boundaries & Limitations

> [!CAUTION]
> **Operational Guardrails on Long Horizons and External Macro Catalysts:**  
> 1. **External Macro Shocks:** Multi-week trends on the Indian market are heavily governed by Reserve Bank of India (RBI) Monetary Policy Committee (MPC) rate decisions, quarterly corporate earnings season (especially Q1-Q4 IT & Banking results), and Union Budget announcements. These external drivers cannot be inferred solely from past candlestick tokens.  
> 2. **Weekly Forecast Horizon:** While Kronos-FT NSE delivers a 46.6% error reduction over the base model on weekly bars, operational position trading must be actively monitored with daily rolling updates.  
> 3. **Focus on Short Horizons:** Restrict automated order dispatch **strictly to 3-5 trading day swing windows** utilizing the rolling forecast discipline.

---

## Disclaimer

> [!WARNING]
> **NOT FINANCIAL OR INVESTMENT ADVICE:**  
> All model outputs, price trajectories, candlestick forecasts, quantitative metrics, and evaluations produced by **Kronos-FT** are provided strictly for academic research, artificial intelligence experimentation, and financial time-series modeling studies. **None of the data, projections, or insights generated constitute formal financial advice, investment recommendations, endorsements, or legal solicitation to buy or sell securities.**  
>
> Trading financial instruments across equities markets (including IDX, NASDAQ, TSE, and NSE) involves substantial volatility and high risk of capital loss. Historical backtesting and past performance do not guarantee future results or profitability.  
>
> Users bear sole and full responsibility for all trading decisions, independent analysis (*Do Your Own Research* / DYOR), and risk management execution. The developers and contributors assume no liability whatsoever for any direct or indirect financial losses, damages, or costs incurred as a result of using this software or these models.

