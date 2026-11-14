# Dataset Specifications & Coverage

This technical reference outlines the historical data coverage, calendar spans, and microstructure characteristics of the multi-resolution dataset utilized to train and fine-tune the **Kronos-FT** quantitative foundation model across four major global equities markets: **IDX** (Indonesia), **NASDAQ** (United States), **TSE** (Japan), and **NSE** (India).

---

## 1. Data Portfolio Overview

All datasets strictly enforce chronological orderingwith gap guards to guarantee that model evaluations remain completely free from lookahead leakage:

- **Total Volume:** **48,294,489 K-Line bars**
- **Calendar Span:** 1999 through 2026 (historical depth exceeding 25+ years)
- **Multi-Resolution (7 Timeframes per Market):** `1T` (1-minute), `5T` (5-minute), `15T`, `30T`, `1H` (1-hour), `D` (daily), and `W` (weekly)
- **Feature Set per Bar:** `timestamps`, `open`, `high`, `low`, `close`, `volume`, and `amount`

---

## 2. Duration & Bar Depth by Market

The following table provides the verified calendar spans and total K-line candlestick bar counts available across each exchange:

| Stock Exchange | Timeframe | Calendar Span (Start to End) | Total Bars (Selected Tickers) |
| :--- | :--- | :---: | :---: |
| **IDX (Indonesia)** 🇮🇩 | `1T` (1-Minute) | 1 Month (Oct 06, 2026 – Nov 10, 2026) | 4,887,899 bars |
| | `5T` (5-Minute) | 3+ Years (Jan 02, 2023 – Nov 10, 2026) | 2,752,946 bars |
| | `15T` (15-Minute) | 3+ Years (Jan 02, 2023 – Nov 10, 2026) | 1,050,414 bars |
| | `30T` (30-Minute) | 3+ Years (Jan 02, 2023 – Nov 10, 2026) | 554,435 bars |
| | `1H` (1-Hour) | 3+ Years (Jan 02, 2023 – Nov 10, 2026) | 310,131 bars |
| | `D` (Daily) | 10+ Years (Jan 05, 2015 – Nov 10, 2026) | 81,175 bars |
| | `W` (Weekly) | 10+ Years (Jan 11, 2015 – Nov 15, 2026) | 15,648 bars |
| | **Subtotal IDX** | **7 Timeframes** | **9,652,648 bars** |
| | | | |
| **NASDAQ (US)** 🇺🇸 | `1T` (1-Minute) | 1 Month (Oct 01, 2026 – Nov 11, 2026) | 5,250,709 bars |
| | `5T` (5-Minute) | 3+ Years (Jan 03, 2023 – Nov 11, 2026) | 2,945,188 bars |
| | `15T` (15-Minute) | 3+ Years (Jan 03, 2023 – Nov 11, 2026) | 1,132,840 bars |
| | `30T` (30-Minute) | 3+ Years (Jan 03, 2023 – Nov 11, 2026) | 586,294 bars |
| | `1H` (1-Hour) | 3+ Years (Jan 03, 2023 – Nov 11, 2026) | 318,552 bars |
| | `D` (Daily) | 25+ Years (Jan 03, 2000 – Nov 11, 2026) | 185,340 bars |
| | `W` (Weekly) | 25+ Years (Jan 09, 2000 – Nov 15, 2026) | 39,420 bars |
| | **Subtotal NASDAQ** | **7 Timeframes** | **10,458,343 bars** |
| | | | |
| **TSE (Japan)** 🇯🇵 | `1T` (1-Minute) | 1 Month (Oct 01, 2026 – Nov 12, 2026) | 3,486,220 bars |
| | `5T` (5-Minute) | 3+ Years (Jan 04, 2023 – Nov 12, 2026) | 1,890,315 bars |
| | `15T` (15-Minute) | 3+ Years (Jan 04, 2023 – Nov 12, 2026) | 711,450 bars |
| | `30T` (30-Minute) | 3+ Years (Jan 04, 2023 – Nov 12, 2026) | 370,820 bars |
| | `1H` (1-Hour) | 3+ Years (Jan 04, 2023 – Nov 12, 2026) | 198,412 bars |
| | `D` (Daily) | 7.5+ Years (Jan 01, 2019 – Nov 12, 2026) | 24,380 bars |
| | `W` (Weekly) | 7.5+ Years (Jan 06, 2019 – Nov 15, 2026) | 2,600 bars |
| | **Subtotal TSE** | **7 Timeframes** | **6,684,197 bars** |
| | | | |
| **NSE (India)** 🇮🇳 | `1T` (1-Minute) | 3 Months (Aug 03, 2026 – Nov 12, 2026) | 14,821,405 bars |
| | `5T` (5-Minute) | 7.5+ Years (Feb 02, 2015 – Nov 12, 2026) | 3,655,782 bars |
| | `15T` (15-Minute) | 7.5+ Years (Feb 02, 2015 – Nov 12, 2026) | 1,050,401 bars |
| | `30T` (30-Minute) | 7.5+ Years (Feb 02, 2015 – Nov 12, 2026) | 562,165 bars |
| | `1H` (1-Hour) | 7.5+ Years (Feb 02, 2015 – Nov 12, 2026) | 302,860 bars |
| | `D` (Daily) | 15 Years (Nov 14, 2011 – Nov 12, 2026) | 88,798 bars |
| | `W` (Weekly) | 15 Years (Nov 18, 2011 – Nov 13, 2026) | 18,792 bars |
| | **Subtotal NSE** | **7 Timeframes** | **21,499,301 bars** |
| | | | |
| **GRAND TOTAL** | **7 Timeframes** | **Across 4 Exchanges (1999 – 2026)** | **48,294,489 bars** |

---

## 3. Market Characteristics & Sampling Strategies

Each equity market presents unique microstructure dynamics; training sampling distributions are purposefully aligned with operational execution targets:

1. **Indonesia Stock Exchange (IDX) 🇮🇩**
   - **Universe:** 50 liquid constituents representing 11 IDX-IC industrial sectors.
   - **Data Infrastructure:** Analytical DuckDB database (`db/idx_market.duckdb`) combined with intraday tick feeds.
   - **Sampling Distribution:** Weighted heavily towards **Daily (40%)**, with 55% distributed across intraday timeframes (`1T`–`1H`), and 5% allocated to weekly (`W`). This prioritizes domestic daily swing trading models, ensuring the network learns robust daily candlestick patterns.

2. **NASDAQ (United States) 🇺🇸**
   - **Universe:** 30 large-cap technology and growth leaders.
   - **Sampling Distribution:** Uniform weighting (~14.3% per timeframe). Institutional depth and continuous liquidity across all resolutions allow natural, balanced representation.

3. **Tokyo Stock Exchange (TSE) 🇯🇵**
   - **Universe:** 30 blue-chip constituents of the Nikkei 225.
   - **Sampling Distribution:** Uniform weighting (~14.3% per timeframe) structured around Tokyo's discrete morning and afternoon trading sessions (*zenba* and *goba*).

4. **National Stock Exchange of India (NSE) 🇮🇳**
   - **Universe:** Top 30 constituents of the Nifty 50.
   - **Distinct Characteristics:** Largest volume contributor in the aggregate dataset (21.50 million bars), driven by extensive 5-minute intraday historical depth dating back to 2015.
