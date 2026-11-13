# Kronos-FT for NASDAQ, NSE, TSE, & IDX

---

Repository untuk fine-tuning model fondasi kuantitatif **Kronos** (*A Foundation Model for the Language of Financial Markets*, Tsinghua University, AAAI 2026) pada bursa saham:

1. **IDX** (Indonesia Stock Exchange) 🇮🇩 - *Active*
2. **NASDAQ** (United States) 🇺🇸 - *Data Ready*
3. **TSE** (Tokyo Stock Exchange, Japan) 🇯🇵 - *Data Ready*
4. **NSE** (National Stock Exchange of India) 🇮🇳 - *Data Ready*

---

## 🌟 Fitur & Spesifikasi Arsitektur

- **Timeframe-Agnostic:** Model dilatih secara terintegrasi melintasi 8 timeframe sekaligus:
  - `1T` (1-menit)
  - `3T` (3-menit)
  - `5T` (5-menit)
  - `15T` (15-menit)
  - `30T` (30-menit)
  - `1H` (1-jam)
  - `D` (Harian)
  - `W` (Mingguan)
- **Zero-Lookahead Temporal Splitting:** Pemisahan dataset (Train 80%, Val 10%, Test 10%) dilakukan secara kronologis ketat.
- **Dual-Model Hub Loading:** Mendukung pemuatan fleksibel baik dari Hugging Face Hub  atau dari direktori lokal (`<market>_tokenizer.safetensors` & `<market>_predictor.safetensors`).

---

## 🚀 Panduan Penggunaan & Pemuatan Model (Inference Quickstart)

Model dan Tokenizer siap digunakan langsung di lokal tanpa perlu melatih ulang. Anda dapat memuatnya melalui beberapa cara yang fleksibel:

### 1. Pemuatan Cepat (Rekomendasi)
```python
from model import KronosPredictor

# Mode Flat Folder (models/idx_tokenizer.safetensors & idx_predictor.safetensors)
predictor = KronosPredictor.from_pretrained("models", exchange="idx", device="cuda:0")

# Atau langsung dari Hugging Face Hub:
predictor = KronosPredictor.from_pretrained("andreiramani/kronos-financial", exchange="idx")
```

### 2. Pemuatan Manual / Fleksibel
```python
from model import KronosTokenizer, Kronos, KronosPredictor

# Cara A: Dari file .safetensors langsung (Single-file mode)
tokenizer = KronosTokenizer.from_pretrained("models/idx_tokenizer.safetensors")
model = Kronos.from_pretrained("models/idx_predictor.safetensors")

# Cara B: Dari subfolder / direktori terpisah (Hugging Face standard)
tokenizer = KronosTokenizer.from_pretrained("outputs/idx/tokenizer/best_model")
model = Kronos.from_pretrained("outputs/idx/predictor/best_model")

# Inisialisasi engine prediksi
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
```

---

## 🛠️ Environment

Dependensi utama:
- `torch>=2.0.0` (dengan akselerasi CUDA)
- `einops>=0.8.0`
- `safetensors>=0.4.0`
- `pandas>=2.0.0`, `numpy>=1.24.0`
- `pyyaml`, `tqdm`, `matplotlib`

---

## 📊 Cakupan Dataset (48.29M Bar K-Line)

Kronos-FT dilatih pada dataset multi-resolusi berskala masif yang mencakup **48.294.489 bar lilin K-line** lintas **4 bursa saham global** (**IDX**, **NASDAQ**, **TSE**, dan **NSE**) pada **7 timeframe** (`1T`, `5T`, `15T`, `30T`, `1H`, `D`, `W`):

* 🇮🇩 **IDX (Indonesia):** 9.65M bar (50 emiten unggulan 11 sektor IDX-IC, 10+ tahun riwayat data).
* 🇺🇸 **NASDAQ (US):** 10.45M bar (30 saham mega-cap teknologi, 25+ tahun data harian sejak tahun 2000).
* 🇯🇵 **TSE (Jepang):** 6.68M bar (30 konstituen Nikkei 225, struktur sesi perdagangan Tokyo).
* 🇮🇳 **NSE (India):** 21.49M bar (30 saham Nifty 50, kedalaman intraday 7.5+ tahun tervalidasi).

> 📖 **Dokumentasi Lengkap Dataset:**  
> Rincian tabel durasi kalender, tanggal mulai/selesai, kedalaman bar per timeframe, dan profil mikrostruktur data dapat dibaca pada [**`DATASET.md`**](docs/DATASET.md).

---

## 👤 Pengembang & Fine-Tuner
- **Lead / Fine-Tuner:** [Andrei Ramani](https://www.linkedin.com/in/andreiramani/)

---

## ⚖️ *Disclaimer*

> [!WARNING]
> **BUKAN NASIHAT INVESTASI RESMI (*FINANCIAL ADVICE DISCLAIMER*):**  
> Seluruh *output*, prediksi harga/arah lilin (*candlestick*), metrik, maupun analisis kuantitatif yang dihasilkan oleh model **Kronos-FT** semata-mata ditujukan untuk keperluan riset, eksperimen kecerdasan buatan (*AI*), dan studi pemodelan deret waktu finansial. **Informasi dan data yang dihasilkan bukan merupakan nasihat keuangan resmi, rekomendasi investasi, ajakan untuk membeli atau menjual instrumen pasar modal, maupun dasar pertimbangan finansial berlisensi.**  
>
> Perdagangan instrumen keuangan di bursa saham (seperti IDX, NASDAQ, TSE, dan NSE) memiliki tingkat volatilitas dan risiko kehilangan modal yang nyata. Kinerja historis (*backtesting* atau *historical performance*) tidak menjamin hasil atau profitabilitas di masa depan. 
>
> Pengguna memikul tanggung jawab penuh secara mandiri atas seluruh keputusan transaksi (*Do Your Own Research* / DYOR) dan pengelolaan risiko (*risk management*). Pengembang dan kontributor dibebaskan dari segala bentuk tuntutan dan tidak bertanggung jawab atas kerugian finansial langsung maupun tidak langsung yang timbul dari penggunaan model ini.

---

## 📜 Lisensi & Atribusi
Base model, original work, and paper:
*Shi et al., "Kronos: A Foundation Model for the Language of Financial Markets", AAAI 2026 / arXiv:2508.02739.*  
Diadaptasi untuk ekosistem kuantitatif pasar modal multi-bursa (IDX, NASDAQ, TSE, NSE).

---