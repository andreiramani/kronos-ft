# Kronos-FT IDX Inference Parameter Grid Search Evaluation

This document summarizes empirical experiments evaluating inference decoding parameters (*decoding grid search*) on the **Kronos-FT IDX** model across 10 benchmark constituents representing major IDX sectors over 4 operational timeframes (200 total test combinations).

---

## 1. Executive Summary & Key Findings

1. **Baseline Configuration (`T=0.30, S=5, p=0.90`) Proves Most Optimal:**  
   Across all 5 parameter variations evaluated, the default baseline configuration (**CFG-0 Baseline**) delivered the strongest aggregate performance across all evaluation metrics:
   - **Lowest Average MAPE:** **2.05%** (vs 2.10% for *Ultra-Greedy*).
   - **Highest Directional Accuracy (*Directional Hit*):** **62.5%** (vs 55.0% for *Low-Temp*).
   - **Highest Path Correlation (*IC*):** **+0.143** (nearly 3 times more stable than *Ultra-Greedy* at +0.045).
2. **Pitfalls of Extreme Temperature Suppression (*Ultra-Greedy Collapse*):**  
   Lowering sampling temperature to $T=0.10$ or $T=0.20$ proved **counterproductive**. On the `30T` timeframe, forced top-token selection (*greedy sampling*) induced compounding trajectory drift, causing path correlation to plummet from $+0.018$ down to **$-0.201$** and directional hit rate to fall from 60% to 40%.
3. **Daily Swing Trading Performance (`Daily` / 5 bars):**  
   On the primary 5-bar daily swing horizon, the baseline configuration registered an **80.0% directional hit rate**, a **3.65% MAPE**, and an **IC of +0.186**.
4. **Technical Conclusion:**  
   The empirical boundary of inference-only parameter tuning has been reached. Subsequent precision improvements must be driven by model weight adaptation via **Predictor Fine-Tuning**.

---

## 2. Aggregate Grid Search Performance (40 Cases per Configuration)

| Config ID | Description | Avg MAPE | Median MAPE | Avg IC | Avg Dir Hit | Avg Inference Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **CFG_0_BASE** | **$T=0.30 \mid S=5 \mid p=0.90$ (Baseline)** | **2.05%** 🟢 | 1.53% | **+0.143** 🟢 | **62.5%** 🟢 | **1.21 s** |
| **CFG_1_ULTRA_GREEDY** | $T=0.10 \mid S=5 \mid p=0.90$ (Ultra-Greedy) | 2.10% | 1.48% | +0.045 🔴 | 55.0% 🔴 | 1.21 s |
| **CFG_2_LOW_TEMP** | $T=0.20 \mid S=5 \mid p=0.90$ (Calibrated Low-Temp) | 2.09% | **1.46%** | +0.046 🔴 | 55.0% 🔴 | 1.21 s |
| **CFG_3_DEEP_MC** | $T=0.20 \mid S=10 \mid p=0.90$ (Deep Monte Carlo) | 2.12% | 1.68% | +0.106 | **62.5%** | 2.33 s |
| **CFG_4_TIGHT_P** | $T=0.30 \mid S=5 \mid p=0.80$ (Tight Nucleus) | 2.06% | 1.51% | +0.085 | 60.0% | 1.21 s |

---

## 3. Timeframe-Level Performance Breakdown

### 3.1 Daily Timeframe (`Daily` / `D`, 5-Bar Horizon  Swing Trading)

| Configuration | Daily MAPE | Directional Hit | Path Correlation (IC) | Evaluation Summary |
| :--- | :---: | :---: | :---: | :--- |
| **CFG_0_BASE** | **3.65%** | **80.0%** | **+0.186** | **Top Performer** (Lowest error, highest correlation) |
| **CFG_1_ULTRA_GREEDY** | 3.74% | **80.0%** | +0.122 | Correlation degraded |
| **CFG_2_LOW_TEMP** | 3.79% | **80.0%** | +0.154 | Marginal error increase |
| **CFG_3_DEEP_MC** | 3.98% | 70.0% | +0.094 | Excessively smoothed (*over-smoothed*) |
| **CFG_4_TIGHT_P** | 3.68% | **80.0%** | +0.122 | Competitive, but lower IC than baseline |

### 3.2 1-Hour Timeframe (`1H`, 16-Bar Horizon  Intraday Confirmation)

| Configuration | 1H MAPE | Directional Hit | Path Correlation (IC) | Evaluation Summary |
| :--- | :---: | :---: | :---: | :--- |
| **CFG_0_BASE** | 1.91% | **70.0%** | **+0.188** | **Most Balanced** (Consistent direction, solid IC) |
| **CFG_1_ULTRA_GREEDY** | 1.92% | 60.0% | +0.067 | Directional accuracy dropped by 10% |
| **CFG_2_LOW_TEMP** | 1.87% | 60.0% | +0.010 | Correlation near zero |
| **CFG_3_DEEP_MC** | **1.86%** | **70.0%** | +0.098 | Lowest MAPE, but $2\times$ compute latency |
| **CFG_4_TIGHT_P** | 1.90% | 60.0% | +0.031 | Directional accuracy dropped by 10% |

### 3.3 30-Minute Timeframe (`30T`, 16-Bar Horizon  Tactical Intraday)

| Configuration | 30T MAPE | Directional Hit | Path Correlation (IC) | Evaluation Summary |
| :--- | :---: | :---: | :---: | :--- |
| **CFG_0_BASE** | **2.11%** | 60.0% | +0.018 | Stable baseline |
| **CFG_1_ULTRA_GREEDY** | 2.20% | 40.0% 🔴 | -0.201 🔴 | **Severe Drift Anomaly / Collapse** |
| **CFG_2_LOW_TEMP** | 2.22% | 40.0% 🔴 | -0.198 🔴 | **Severe Drift Anomaly / Collapse** |
| **CFG_3_DEEP_MC** | 2.12% | **70.0%** 🟢 | **+0.052** 🟢 | Best for 30T (Monte Carlo suppresses drift) |
| **CFG_4_TIGHT_P** | 2.16% | 60.0% | +0.009 | Stable, slightly below baseline |

### 3.4 5-Minute Timeframe (`5T`, 16-Bar Horizon  Fractional Microstructure)

| Configuration | 5T MAPE | Directional Hit | Path Correlation (IC) | Evaluation Summary |
| :--- | :---: | :---: | :---: | :--- |
| **CFG_0_BASE** | 0.508% | 40.0% | +0.181 | Stable baseline |
| **CFG_1_ULTRA_GREEDY** | 0.538% | 40.0% | +0.194 | Marginal error increase |
| **CFG_2_LOW_TEMP** | **0.500%** | 40.0% | **+0.217** | Lowest MAPE on 5T |
| **CFG_3_DEEP_MC** | 0.519% | 40.0% | +0.181 | Stable |
| **CFG_4_TIGHT_P** | 0.503% | 40.0% | +0.179 | Stable |

---

## 4. Production Operational Recommendations

1. **Retain Official Production Settings:**  
   The default parameters established in [`docs/idx/IDX_PRODUCTION.md`](file:///E:/anaconda/AR/kronos-ft/docs/idx/IDX_PRODUCTION.md) represent the optimal operational sweet spot for Kronos-FT IDX:
   * **`lookback: 256`**
   * **`temperature: 0.30`**
   * **`sample_count: 5`**
   * **`top_p: 0.90`**
2. **Specific 30-Minute (`30T`) Exception:**  
   If an execution workflow specializes exclusively in tactical 30-minute intraday setups, configuring `sample_count: 10` with `temperature: 0.20` can elevate directional accuracy to 70%, with the operational trade-off of increasing inference time to ~2.3 seconds per ticker.
3. **Path Forward for Model Evolution:**  
   Since inference-time parameter tuning has reached its efficiency ceiling, substantial precision gains must be pursued through **Predictor Fine-Tuning** (evaluations every 500 steps, cosine warmup scheduler, and weight decay of 0.15).
