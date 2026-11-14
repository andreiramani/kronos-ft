# Kronos NASDAQ Fine-Tuning: Ablation & Iteration Log

Author: Andrei Ramani  
Market: NASDAQ (United States)  
Hardware: NVIDIA GeForce RTX 3050 Laptop GPU 6GB, Windows 11  

## Summary of Iteration Rounds

### Round 1: v1 Baseline Failure Audit
- **Issue:** v1 fine-tuning produced severe degradation on NASDAQ intraday and daily.
- **Root Causes Diagnosed:**
  1. Pre/post-market session noise mixed into standard timeframes causing non-uniform time grids.
  2. Fine-tuning the tokenizer shifted the BSQ codebook embeddings, causing representation drift.
  3. Greedy sampling temperature (T=0.15) caused flatline autoregressive drift.

### Round 2: Data Engineering & Run A (Strict RTH + Frozen Base Tokenizer)
- Enforced strict Regular Trading Hours (09:30 - 16:00 America/New_York).
- Zero-leakage temporal splits with gap = 288 bars (256 lookback + 32 predict).
- Frozen Base Tokenizer (`pretrained/Kronos-Tokenizer-base`).
- Validation CE Loss dropped from 2.2658 to 2.1727.
- **Daily Directional Accuracy:** Jumped to **70.0% vs Base 30.0%**, MAPE dropped to 4.50%.
- **Intraday Gap:** 1H directional hit was 40% vs Base 80%.

### Round 3: Run B (Expanded Sample Exposure 25k/epoch + Cosine Warmup)
- Increased exposure to 25,000 samples/epoch with 500-step linear warmup.
- Result: Early stopping triggered after Epoch 2 as full fine-tuning started to overfit (train loss 1.86 vs val loss 2.36).

### Round 4: Run C (Task-Conditioned LoRA + Auxiliary Direction Loss)
- 100% frozen base model + LoRA rank 16 on attention projections (1.33% trainable params).
- Integrated differentiable expected-price decoder with BCE direction loss.
- Peak VRAM: 1.50 GB (super lightweight).
- Daily hit: 60%, but 1H directional hit was 50%.

### Round 5: Checkpoint Interpolation (Model Soup)
- Tested linear weight interpolation: $\theta = \alpha \theta_{\text{Run A}} + (1 - \alpha) \theta_{\text{Base}}$.
- **Soup A50 ($\alpha=0.50$):** Reached the Pareto frontier:
  - Daily Dir Hit: **60.0% vs Base 30.0%** (2x Base)
  - 1H Dir Hit: **70.0% vs Base 80.0%**
  - Overall Return MAE: **2.499% vs Base 2.920%**
  - Overall MAPE: **2.16% vs Base 2.53%**
- **Verdict:** Promoted **Soup A50** to production `models/`.