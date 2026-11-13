<p align="center">
  <img src="docs/assets/kronos-ft_logo.webp" alt="Kronos-FT Logo" width="600">
</p>

# Kronos-FT for NASDAQ, NSE, TSE, & IDX

---

Fine-tuning the quantitative foundation model **Kronos** (*A Foundation Model for the Language of Financial Markets*, Tsinghua University, AAAI 2026) tailored specifically across equity markets:

1. **IDX** (Indonesia Stock Exchange) 🇮🇩 - *Active*
2. **NASDAQ** (United States) 🇺🇸 - *Data Ready*
3. **TSE** (Tokyo Stock Exchange, Japan) 🇯🇵 - *Data Ready*
4. **NSE** (National Stock Exchange of India) 🇮🇳 - *Data Ready*

---

## 🌟 Key Features & Architecture Specifications

- **Timeframe-Agnostic:** The model is trained in a unified manner across 8 timeframes simultaneously:
  - `1T` (1-minute)
  - `3T` (3-minute)
  - `5T` (5-minute)
  - `15T` (15-minute)
  - `30T` (30-minute)
  - `1H` (1-hour)
  - `D` (Daily)
  - `W` (Weekly)
- **Zero-Lookahead Temporal Splitting:** Dataset partitioning (Train 80%, Val 10%, Test 10%) strictly enforces chronological ordering with temporal gap guards to prevent future lookahead leakage.
- **Dual-Model Hub Loading:** Supports flexible loading from either the Hugging Face Hub or local directories.

---

## 🚀 Quickstart & Model Inference

Model weights and tokenizers are pre-calibrated and ready for local inference without retraining. You can load them via multiple flexible options (example command below for IDX market):

### 1. Fast One-Line Loading (Recommended)
```python
from model import KronosPredictor

# Flat Directory Mode (models/idx_tokenizer.safetensors & idx_predictor.safetensors)
predictor = KronosPredictor.from_pretrained("models", exchange="idx", device="cuda:0")

# Or directly from the Hugging Face Hub:
predictor = KronosPredictor.from_pretrained("andreiramani/kronos-ft", exchange="idx")
```

### 2. Manual / Granular Loading
```python
from model import KronosTokenizer, Kronos, KronosPredictor

# Option A: Directly from single-file .safetensors checkpoints
tokenizer = KronosTokenizer.from_pretrained("models/idx_tokenizer.safetensors")
model = Kronos.from_pretrained("models/idx_predictor.safetensors")

# Option B: From separate directory / Hugging Face layout
tokenizer = KronosTokenizer.from_pretrained("outputs/idx/tokenizer/best_model")
model = Kronos.from_pretrained("outputs/idx/predictor/best_model")

# Initialize inference predictor engine
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
```

---

## 🛠️ Environment Requirements

Key dependencies:
- `torch>=2.0.0` (with CUDA acceleration)
- `einops>=0.8.0`
- `safetensors>=0.4.0`
- `pandas>=2.0.0`, `numpy>=1.24.0`
- `pyyaml`, `tqdm`, `matplotlib`

---

## 📊 Dataset Coverage

Kronos-FT was trained on a massive multi-resolution dataset comprising **48,294,489 K-line candlestick bars** across **4 global stock exchanges** (**IDX**, **NASDAQ**, **TSE**, and **NSE**) covering **7 timeframes** (`1T`, `5T`, `15T`, `30T`, `1H`, `D`, `W`):

* 🇮🇩 **IDX (Indonesia):** 9.65M bars (50 liquid constituents across 11 IDX-IC sectors, 10+ years of historical data).
* 🇺🇸 **NASDAQ (US):** 10.45M bars (30 mega-cap tech equities, 25+ years of daily data since 2000).
* 🇯🇵 **TSE (Japan):** 6.68M bars (30 Nikkei 225 constituents, Tokyo trading session market structure).
* 🇮🇳 **NSE (India):** 21.49M bars (30 Nifty 50 equities, 7.5+ years of verified intraday depth).

> 📖 **Comprehensive Dataset Documentation:**  
> Detailed tables covering date ranges, calendar durations, bar counts per timeframe, and microstructure characteristics are available in [**`DATASET.md`**](docs/DATASET.md).

---

## 👤 Developer & Fine-Tuner
- **Lead / Fine-Tuner:** [Andrei Ramani](https://www.linkedin.com/in/andreiramani/)

---

## ⚖️ Disclaimer

> [!WARNING]
> **NOT FINANCIAL OR INVESTMENT ADVICE:**  
> All model outputs, price trajectories, candlestick forecasts, quantitative metrics, and evaluations produced by **Kronos-FT** are provided strictly for academic research, artificial intelligence experimentation, and financial time-series modeling studies. **None of the data, projections, or insights generated constitute formal financial advice, investment recommendations, endorsements, or legal solicitation to buy or sell securities.**  
>
> Trading financial instruments across equities markets (including IDX, NASDAQ, TSE, and NSE) involves substantial volatility and high risk of capital loss. Historical backtesting and past performance do not guarantee future results or profitability.  
>
> Users bear sole and full responsibility for all trading decisions, independent analysis (*Do Your Own Research* / DYOR), and risk management execution. The developers and contributors assume no liability whatsoever for any direct or indirect financial losses, damages, or costs incurred as a result of using this software or these models.

---

## 📜 License & Attribution
Base model, original work, and paper:
*Shi et al., "Kronos: A Foundation Model for the Language of Financial Markets", AAAI 2026 / arXiv:2508.02739.*  
Adapted for quantitative financial multi-exchange ecosystems (IDX, NASDAQ, TSE, NSE).
