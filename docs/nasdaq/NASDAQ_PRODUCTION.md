# Kronos-FT NASDAQ Production Playbook (Institutional Mock Trading Implementation)

This guide outlines best practices, execution architecture, and operational risk controls for deploying the **Kronos-FT NASDAQ** foundation model into mock production quantitative workflows across United States equities.

---

## 1. Model Role in the Production Architecture

Kronos-FT NASDAQ is an autoregressive time-series foundation model designed to process non-stationary financial candlestick dynamics. Within an institutional quantitative trading framework, the model functions as a **directional confirmation filter and forward momentum validator**, rather than an unconstrained autonomous black-box executor:

- **Core Operational Focus:** Supports multi-day **Daily Swing Trading (`Daily` / 3-5 trading day horizon)** and tactical **intraday execution timing (`1H` / 16 bars)**.
- **Signal Confluence:** Projected trajectories validate whether multi-day trend momentum aligns with primary quantitative screening rules (e.g., trend breakout, volatility contraction, institutional liquidity sweeps) prior to dispatching orders to broker APIs.

---

## 2. Locked Production Specifications

Based on extensive empirical evaluations across 40 out-of-sample benchmark scenarios, the following production inference parameters are locked to maintain elevated directional accuracy ($\ge 60\%$) while minimizing return forecast deviations (Return MAE $\le 2.50\%$):

| Parameter | Production Value | Architectural Justification |
| :--- | :---: | :--- |
| **`lookback_window`** | **256 bars** | Optimal context memory; preserves positive path correlation (IC) without inducing attention dilution from stale macro regimes. |
| **`pred_len` (Daily)** | **5 days** | Exactly one calendar trading week; represents a complete institutional swing cycle and prevents compounding trajectory drift. |
| **`pred_len` (Intraday)** | **16 bars** | Captures 16 trading hours (approx. 2.5 full RTH sessions); ideal for pullback timing and entry optimization. |
| **`temperature_daily`** | **0.30** | Calibrated sampling temperature; models multi-day equity volatility without flatlining or excessive stochastic dispersion. |
| **`temperature_intraday`**| **0.15** | Low-entropy sampling; suppresses high-frequency microstructure noise for hourly execution bars. |
| **`sample_count`** | **5** | Averages across 5 Monte Carlo projection paths to stabilize price level estimates against single-sample variance. |
| **`top_p`** | **0.90** | Standard nucleus sampling; eliminates improbable token tails while retaining legitimate volatility spikes. |

---

## 3. Operational Discipline: EOD Rolling Forecast

The foundational discipline for deploying autoregressive foundation models in production: **never allow forecasts to become stale**.

```
┌────────────────────────────────────────────────────────┐
│ 1. POST-MARKET DATA INGESTION (16:30 EST)              │
│ - Collect official RTH closing prints (09:30-16:00)    │
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
│   (Kronos Direction == UP) AND (Screening Score >= 70) │
└────────────────────────────────────────────────────────┘
```

1. **Daily Execution Schedule (16:30 EST / Post-Market Close):**
   - As soon as the NASDAQ Regular Trading Hours (RTH) close and settlement prints finalize, ingest official adjusted OHLCV data into the database.
2. **Extract Trailing 256 Bars:**
   - Retrieve the most recent 256 bars for active universe constituents, then execute `KronosPredictor` to forecast the upcoming **5 trading days**.
3. **Order Confluence Rule:**
   - A long swing order may only be dispatched when both conditions are satisfied concurrently:
     $$\text{Kronos Directional Forecast } (P_{t+5} > P_t) \quad \text{AND} \quad \text{Quantitative Strategy Score} \ge 70$$
4. **Shift Context Window Daily (Rolling Updates):**
   - After each session concludes, slide the 256-bar memory window forward by one step ($T+1$). This ensures all inferences are conditioned on the freshest market data, keeping realized daily MAPE tightly bounded below **5.0%**.

---

## 4. Alignment with US Equity & NASDAQ Microstructure Regulations

AI foundation models process quantitative time-series patterns and do not natively account for US market trading mechanics. The execution engine must strictly enforce regulatory and market microstructure rules:

1. **Strict Regular Trading Hours (RTH) Enforcement:**
   - Model inference must **strictly consume RTH data (09:30 - 16:00 America/New_York)**.
   - Do **NOT** feed unadjusted pre-market (04:00-09:30) or post-market (16:00-20:00) prints into the 256-bar context window, as low liquidity and wide bid-ask spreads inject severe token quantization distortion.
2. **Earnings Announcement Blackout Window:**
   - **Mandatory Filter:** Prohibit establishing new swing positions within the $[\text{EarningsDate}-2\text{d}, \text{EarningsDate}+1\text{d}]$ window for any equity with an upcoming quarterly earnings report.
   - Foundation models cannot anticipate binary fundamental shocks (e.g., revenue surprises, guidance revisions).
3. **Volatility-Adjusted Position Sizing (ATR Parity):**
   - High-beta constituents (e.g., **NVDA**, **TSLA**) have average 5-day historical volatility of 10%-14%, compared to defensive staples like **COST** (1.8%-2.5%).
   - Position sizing must be scaled inversely to volatility ($w_i \propto 1 / \text{ATR}_{14}$), preventing high-beta names from dominating portfolio variance.
4. **Anti-Chasing Limit at Open:**
   - Prohibit market buy execution if a stock gaps up at the open by more than $0.5 \times \text{ATR}_{14}$ above the previous close. Wait for an intraday 1-hour pullback.

---

## 5. Usage Boundaries & Hard Limitations

> [!CAUTION]
> **1. Do NOT use this model for Weekly bar forecasting or horizons $> 5$ days!**  
> Volatility dispersion ($\sigma\sqrt{\Delta t}$) expands significantly beyond 5 trading days. Multi-week trends are heavily governed by macroeconomic shifts, Federal Reserve interest rate announcements, and geopolitical events that cannot be deduced from price history alone.
>
> **2. Do NOT use this model as a standalone 5-minute / 30-minute high-frequency scalper!**  
> While the model's price tracking error on 5-minute bars is tight (MAPE 0.23%), directional accuracy on high-frequency noise drops to ~40%. The model is trained to recognize institutional swing cycles, not tick-level order book imbalances.
>
> **3. Strict Stop-Loss Discipline:**  
> Always anchor stop-loss orders to multi-day Daily Swing Lows/Highs (using a $1.5 \times \text{ATR}_{14}$ buffer), never to tight intraday micro-levels which are vulnerable to market-maker liquidity sweeps.
