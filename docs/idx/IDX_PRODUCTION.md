# Kronos-FT IDX Production Playbook (Mock Live Trading Implementation)

This guide outlines best practices for deploying the **Kronos-FT IDX** quantitative foundation model into mock trading workflows across the Indonesia Stock Exchange (IDX / BEI).

---

## 1. Model Role in the Production Architecture

Kronos-FT IDX is not designed to make standalone execution decisions autonomously. Within any quantitative framework, the model functions as a **directional confirmation filter and forward momentum validator**:

- **Core Focus:** Supports decision-making for **Daily Swing Trading (`Daily` / 3-5 trading day horizon)** and **tactical intraday execution (`1T`-`1H`)**.
- **Signal Confluence:** The model's projected trajectories are utilized to verify whether forward momentum aligns with the qIDX Dual-Engine quantitative algorithmic scoring prior to dispatching buy orders to the market.

---

## 2. Locked Production Specifications

Based on extensive empirical evaluations, utilize the following parameters to maintain elevated directional accuracy ($\ge 70\%$) while minimizing price deviations (average daily MAPE $\le 3.5\%$):

| Parameter | Production Value | Architectural Justification |
| :--- | :---: | :--- |
| **`lookback_window`** | **256 bars** | Optimal context length; preserves positive path correlation (IC) without inducing attention dilution from stale macro regimes. |
| **`pred_len` (Daily)** | **3 - 5 days** | Optimal horizon for a single swing cycle; prevents compounding trajectory drift associated with over-extended horizons. |
| **`pred_len` (Intraday)** | **16 bars** | Balanced window for tracking momentum continuation or intraday support/resistance bounces. |
| **`temperature`** | **0.3** | Low sampling temperature to suppress stochastic noise, ensuring the model consistently decodes highest-probability token sequences. |
| **`sample_count`** | **5** | Averages across 5 Monte Carlo projection paths to stabilize price level estimates against single-sample variance. |

---

## 3. Operational Discipline: EOD Rolling Forecast

The core principle when deploying autoregressive time-series foundation models: **never allow forecasts to become stale**.

1. **Daily Execution Schedule (16:15 WIB / Post-Market Close):**
   - As soon as the exchange pre-closing session concludes, ingest official end-of-day market data (`adj_close`, volume, and transaction value) into the database.
2. **Extract the Trailing 256 Bars:**
   - Retrieve the most recent 256 bars for candidate tickers, then execute the inference engine to forecast the upcoming **3-5 trading days**.
3. **Order Confluence Rule:**
   - A buy order may only be executed when both conditions are satisfied concurrently:
     $$\text{Kronos Directional Forecast } (P_{t+5} > P_t) \quad \text{AND} \quad \text{qIDX Swing Signal Score} \ge 70$$
4. **Shift Context Window Daily (Rolling Updates):**
   - After each trading day concludes, shift the 256-bar memory window forward by one step ($T+1$). This ensures all inferences are conditioned on the freshest market data, keeping realized daily MAPE tightly bounded between **1.5% and 3.5%**.

---

## 4. Alignment with IDX Market Microstructure Regulations

AI foundation models process quantitative time-series patterns and do not natively account for local market constraints. The execution engine must strictly enforce regulatory rules:

1. **Official IDX Price Tick Rounding:**
   - Take Profit (TP) targets must always be rounded up (*ceil tick*).
   - Stop Loss (SL) triggers must always be rounded down (*floor tick*).
2. **Anti-Chasing Limit:**
   - Maximum opening market buy price limit: $P_{\text{limit}} \le P_{\text{close}} + 0.5 \times \text{ATR}_{14}$. Prohibit order execution if the stock gaps up beyond this threshold at the open.
3. **Dividend and Listing Status Safeguards:**
   - **Dividend Blackout Window:** Avoid establishing long positions during the $[\text{CumDate}-3\text{d}, \text{ExDate}+1\text{d}]$ window for equities with an estimated dividend yield $\ge 3.5\%$ to prevent post-cum-date dividend trap drawdowns.
   - **Total Exclusion Filter:** Stocks placed on the Special Monitoring Board (FCA / Full Call Auction) and tickers under active UMA (Unusual Market Activity) status or trading suspensions **must be 100% blocked**.

---

## 5. Usage Boundaries & Limitations

> [!CAUTION]
> **Do not utilize the model for weekly candlestick forecasting (weekly bars / horizon > 5 trading days)!**  
> 1. **Volatility Dispersion ($\sigma\sqrt{\Delta t}$):** Over a 1-month horizon (4 weekly bars / ~20 trading days), cumulative price uncertainty expands by approximately 4.5 times ($\sigma\sqrt{\Delta t}$). Candlestick bodies and wicks are prone to severe structural distortion.  
> 2. **External Fundamental Drivers:** Multi-week trends are heavily governed by earnings announcements, dividend distributions, monetary policy shifts, and macroeconomic data that cannot be inferred solely from past price action.  
> 3. **Focus on Short Horizons:** Restrict operational use **strictly to 3-5 trading day swing windows** utilizing the rolling forecast discipline.
