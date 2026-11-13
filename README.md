# Kronos-FT for NASDAQ, NSE, TSE, & IDX

---

Repository untuk fine-tuning model fondasi kuantitatif **Kronos** (*A Foundation Model for the Language of Financial Markets*, Tsinghua University, AAAI 2026) secara spesifik per bursa saham:

1. **IDX** (Indonesia Stock Exchange) 🇮🇩 - *Active*
2. **NASDAQ** (United States) 🇺🇸 - *Data Ready*
3. **TSE** (Tokyo Stock Exchange, Japan) 🇯🇵 - *Data Ready*
4. **NSE** (National Stock Exchange of India) 🇮🇳 - *Data Ready*

---

## ⚠️ Batasan & Peringatan Risiko Model (Operational Caution)

> [!CAUTION]
> **JANGAN GUNAKAN KRONOS-FT IDX UNTUK PREDIKSI CANDLE MINGGUAN (WEEKLY CANDLESTICK / > 5 HARI BURSA)!**  
> Model ini dikurasi dan dilatih secara spesifik untuk **eksekusi taktis Candle Intraday (`1T`–`1H`) dan Candle Harian (`Daily` / 3–5 candle)** dengan alokasi bobot 95%.  
> Pada skala mingguan/bulanan, difusi volatilitas ($\sigma \sqrt{\Delta t}$) melebar hingga $4.5\times$ lipat dan memicu distorsi anatomi candlestick (*body & shadow deformation*). Selain itu, pergerakan candle mingguan didominasi aksi korporasi dan variabel makroekonomi yang tidak terliput oleh data lilin masa lalu.  
> **Mandat Operasional:** Dilarang menggunakan candle mingguan sebagai sinyal eksekusi/stop loss. Gunakan model hanya untuk proyeksi candle harian **3–5 hari bursa** dengan prinsip *Rolling Forecast* (perbarui bar EOD setiap sore pukul 16:15 WIB). Detail spesifikasi dapat dibaca pada [`docs/IDX_PRODUCTION.md`](docs/IDX_PRODUCTION.md).

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
- **Dual-Model Hub Loading:** Mendukung pemuatan fleksibel baik dari Hugging Face Hub subfolder (`subfolder="idx/predictor"`), direktori lokal terpisah, maupun **folder tunggal flat (`models/`)** dengan nama file spesifik bursa bebas tabrakan (`idx_tokenizer.safetensors` & `idx_predictor.safetensors`).

---

## 🚀 Panduan Penggunaan & Pemuatan Model (Inference Quickstart)

Model dan Tokenizer siap digunakan langsung di lokal tanpa perlu melatih ulang. Anda dapat memuatnya melalui beberapa cara yang fleksibel:

### 1. Pemuatan Cepat Satu Baris (Rekomendasi)
```python
from model import KronosPredictor

# Mode Flat Folder (models/idx_tokenizer.safetensors & idx_predictor.safetensors)
predictor = KronosPredictor.from_pretrained("models", exchange="idx", device="cuda:0")

# Atau langsung dari Hugging Face Hub:
predictor = KronosPredictor.from_pretrained("andreiramani/kronos-financial", exchange="idx")
```

### 2. Pemuatan Terpisah (Manual / Fleksibel)
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

### 3. Pengemasan Otomatis Checkpoint (`scripts/package_models.py`)
Untuk mengonversi checkpoint lokal ke format folder tunggal `models/`:
```bash
python scripts/package_models.py --exchange idx
```
Output tersimpan rapi tanpa tabrakan nama:
* `models/idx_tokenizer.safetensors` + `models/idx_tokenizer.json`
* `models/idx_predictor.safetensors` + `models/idx_predictor.json`

---

## 🛠️ Persyaratan Lingkungan

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
> Rincian tabel durasi kalender, tanggal mulai/selesai, kedalaman bar per timeframe, dan profil mikrostruktur data dapat dibaca pada [**`docs/DATASET.md`**](docs/DATASET.md).

---

## 👤 Pengembang & Fine-Tuner
- **Lead / Fine-Tuner:** [Andrei Ramani](https://www.linkedin.com/in/andreiramani/)

---

## 📜 Lisensi & Atribusi
Base model, original work, and paper:
*Shi et al., "Kronos: A Foundation Model for the Language of Financial Markets", AAAI 2026 / arXiv:2508.02739.*  
Diadaptasi untuk ekosistem kuantitatif pasar modal multi-bursa (IDX, NASDAQ, TSE, NSE).

---

## ⚖️ *Disclaimer*

> [!WARNING]
> **BUKAN NASIHAT INVESTASI RESMI (*FINANCIAL ADVICE DISCLAIMER*):**  
> Seluruh *output*, prediksi harga/arah lilin (*candlestick*), metrik, maupun analisis kuantitatif yang dihasilkan oleh model **Kronos-FT** semata-mata ditujukan untuk keperluan riset, eksperimen kecerdasan buatan (*AI*), dan studi pemodelan deret waktu finansial. **Informasi dan data yang dihasilkan bukan merupakan nasihat keuangan resmi, rekomendasi investasi, ajakan untuk membeli atau menjual instrumen pasar modal, maupun dasar pertimbangan finansial berlisensi.**  
>
> Perdagangan instrumen keuangan di bursa saham (seperti IDX, NASDAQ, TSE, dan NSE) memiliki tingkat volatilitas dan risiko kehilangan modal yang nyata. Kinerja historis (*backtesting* atau *historical performance*) tidak menjamin hasil atau profitabilitas di masa depan. 
>
> Pengguna memikul tanggung jawab penuh secara mandiri atas seluruh keputusan transaksi (*Do Your Own Research* / DYOR) dan pengelolaan risiko (*risk management*). Pengembang dan kontributor dibebaskan dari segala bentuk tuntutan dan tidak bertanggung jawab atas kerugian finansial langsung maupun tidak langsung yang timbul dari penggunaan model ini.

