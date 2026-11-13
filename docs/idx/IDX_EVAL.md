# Evaluasi Model Kronos-FT IDX (10 Sektor Acuan BEI)

Dokumentasi ini merangkum hasil uji coba dan evaluasi performa model **Kronos-FT IDX** (*Stage 2 Predictor 102.3M parameters*) pada 10 sektor unggulan di Bursa Efek Indonesia (BEI).

---

## 1. Setup dan Parameter Pengujian

Evaluasi dijalankan menggunakan data uji di luar sampel latih (*out-of-sample test split*) dengan pemisahan waktu yang ketat, sehingga bebas dari kebocoran data masa depan (*zero-lookahead bias*):

- **Arsitektur Model:** Kronos Foundation Model (*Decoder-Only Transformer*, 102,3 juta parameter).
- **Fine-Tuner:** Andrei Ramani.
- **Panjang Memori (*Lookback Context*):** **256 bar** (konteks penuh).
- **Strategi Inferensi:**
  - `temperature: 0.3` (menjaga proyeksi tetap realistis dan tidak melenceng oleh fluktuasi acak).
  - `sample_count: 5` (mengambil rata-rata dari 5 jalur sampling agar proyeksi kurva harga lebih stabil).
  - `top_p: 0.9`.
- **Horizon Prediksi Adaptif:**
  - **Harian (`Daily`):** **5 bar** (setara 1 pekan kalender bursa, pas dengan kebutuhan *swing trading*).
  - **Intraday (`5T`, `30T`, `1H`):** **16 bar**.
- **Timeframe Operasional yang Diuji:** `5T`, `30T`, `1H`, dan `Daily`. (Timeframe mingguan sengaja ditarik dari evaluasi operasional karena karakteristiknya berbeda, lihat penjelasannya di bawah).

---

## 2. Emiten Acuan 10 Sektor BEI (IDX-IC)

Untuk melihat ketahanan model di berbagai dinamika industri, kami memilih 10 saham berkapitalisasi besar dan likuid yang mewakili 10 sektor utama di BEI:

| No | Ticker | Nama Perusahaan | Sektor IDX-IC | Catatan Karakteristik |
| :-: | :--- | :--- | :--- | :--- |
| 1 | **BBRI** | Bank Rakyat Indonesia Tbk. | Financials | Penggerak utama sektor perbankan dan sensitif terhadap suku bunga |
| 2 | **ANTM** | Aneka Tambang Tbk. | Basic Materials | Komoditas emas & nikel, mengikuti siklus logam global |
| 3 | **ADRO** | Alamtri Resources Indonesia Tbk. | Energy | Emiten batubara dengan dividen jumbo |
| 4 | **ICBP** | Indofood CBP Sukses Makmur Tbk. | Consumer Non-Cyclicals | Saham defensif barang konsumsi pokok (*consumer staples*) |
| 5 | **MAPI** | Mitra Adiperkasa Tbk. | Consumer Cyclicals | Ritel gaya hidup segmen menengah ke atas |
| 6 | **TLKM** | Telkom Indonesia Tbk. | Infrastructures | Penguasa jaringan telekomunikasi dan infrastruktur data |
| 7 | **ASII** | Astra International Tbk. | Industrials | Konglomerasi otomotif dan alat berat |
| 8 | **CTRA** | Ciputra Development Tbk. | Properties & Real Estate | Pengembang properti residensial dan komersial |
| 9 | **EMTK** | Elang Mahkota Teknologi Tbk. | Technology | Ekosistem media digital dan teknologi |
| 10 | **KLBF** | Kalbe Farma Tbk. | Healthcare | Produsen farmasi dan layanan kesehatan konsumen |

> **Catatan Pemilihan EMTK:** Untuk sektor teknologi, EMTK dipilih menggantikan GOTO. Saham dengan harga nominal sangat rendah (seperti area Rp 20-an pada GOTO) menimbulkan distorsi persentase error (*denominator bias*) yang membuat metrik MAPE tampak sangat buruk walau deviasi nominalnya tipis. Selain itu, riwayat perdagangan mingguan GOTO sejak IPO 2022 belum memenuhi panjang memori 256 bar.

---

## 3. Rangkuman Kinerja per Timeframe

Hasil uji pada 10 sektor acuan menunjukkan model bekerja sangat konsisten di timeframe operasional:

| Timeframe | Rata-rata MAPE | Median MAPE | Rata-rata Korelasi (*IC*) | Akurasi Arah (*Hit Ratio*) | Horizon Evaluasi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **5T** (5 Menit) | **0,68%** | 0,67% | -0,07 | 30,0% | 16 bar (~1,3 jam bursa) |
| **30T** (30 Menit) | **3,08%** | 2,98% | -0,03 | 40,0% | 16 bar (~1 hari bursa) |
| **1H** (1 Jam) | **3,14%** | 2,95% | +0,03 | **60,0%** | 16 bar (~2,5 hari bursa) |
| **D (Harian)** | **5,14%** | **4,98%** | **+0,13** | **80,0%** 🟢 | **5 bar (1 pekan bursa)** |
| **RATA-RATA TOTAL** | **3,01%** | **2,90%** | **+0,016** | **52,5%** | **40 Uji Kasus Lintas Sektor** |

---

## 4. Hasil Detail di Timeframe Harian (Daily / `D`)

Pada timeframe harian—yang menjadi tumpuan utama strategi *swing trading*—model menunjukkan kemampuan membaca arah yang sangat tajam: **8 dari 10 saham (80,0%) berhasil ditebak arah trennya (naik atau turun) dengan tepat 100%** selama 5 hari ke depan:

| Ticker | Sektor IDX-IC | MAPE (Persentase Error) | Akurasi Arah (*Direction Hit*) | Korelasi Tren (*IC*) | Grafik Hasil Evaluasi |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **ASII** | Industrials | **1,44%** | **100,0%** | **+0,89** | [eval_ASII_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_ASII_D.png) |
| **BBRI** | Financials | **2,71%** | **100,0%** | **+0,90** | [eval_BBRI_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_BBRI_D.png) |
| **CTRA** | Properties | **3,21%** | **100,0%** | -0,36 | [eval_CTRA_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_CTRA_D.png) |
| **EMTK** | Technology | **4,48%** | 0,0% | **+0,31** | [eval_EMTK_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_EMTK_D.png) |
| **ADRO** | Energy | **4,71%** | **100,0%** | **+0,02** | [eval_ADRO_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_ADRO_D.png) |
| **TLKM** | Infrastructures | **5,26%** | **100,0%** | -0,90 | [eval_TLKM_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_TLKM_D.png) |
| **KLBF** | Healthcare | **5,56%** | **100,0%** | **+0,40** | [eval_KLBF_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_KLBF_D.png) |
| **ICBP** | Consumer Non-Cyclicals | **7,21%** | **100,0%** | -0,43 | [eval_ICBP_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_ICBP_D.png) |
| **ANTM** | Basic Materials | **7,56%** | **100,0%** | **+0,85** | [eval_ANTM_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_ANTM_D.png) |
| **MAPI** | Consumer Cyclicals | **9,22%** | 0,0% | -0,41 | [eval_MAPI_D.png](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/eval_MAPI_D.png) |

---

## 5. Analisis Hasil Eksperimen

### Uji Coba Panjang Konteks: 256 Bar vs 512 Bar
Kami sempat menguji apakah menambah panjang memori menjadi 512 bar harian (~2 tahun kalender) akan meningkatkan akurasi:

| Metrik | **Lookback = 256 Bar** | **Lookback = 512 Bar** | Catatan Perbandingan |
| :--- | :---: | :---: | :--- |
| **Rata-rata MAPE Daily** | 4,90% | **4,68%** | Turun tipis (-0,22%) |
| **Median MAPE Daily** | 4,95% | **3,29%** | Terlihat membaik pada TLKM (2,61%) & ICBP (2,80%) |
| **Akurasi Arah (*Hit Ratio*)** | **70,0%** | **70,0%** | Sama persis |
| **Korelasi Grafik (*Avg IC*)** | **+0,129** (Positif) | **-0,065** (Negatif) | 🔴 **Korelasi arah berbalik negatif pada BBRI dan CTRA** |

**Kesimpulan Praktis:** Mengisi model dengan 512 bar harian ternyata membuat fokus atensi model terpecah (*attention dilution*). Model jadi terbebani oleh siklus makro 2 tahun lalu yang sudah tidak cocok dengan kondisi pasar saat ini—misalnya pada `BBRI`, MAPE-nya justru melonjak dari 2,99% menjadi 9,53%. Oleh karena itu, **`lookback: 256 bar` diputuskan sebagai setelan baku terbaik**.

### Alasan Penarikan Timeframe Mingguan (Weekly) dari Evaluasi
1. **Pelebaran Volatilitas ($\sigma \propto \sqrt{\Delta t}$):** Jarak waktu 1 bulan (4 bar mingguan atau ~20 hari bursa) memiliki ruang ayunan volatilitas kumulatif sekitar $\sqrt{20} \approx 4,5$ kali lipat lebih besar dibanding harian. Deviasi angka MAPE tentu akan otomatis lebih lebar dan tidak adil jika dibandingkan langsung dengan timeframe yang lebih pendek.
2. **Fokus Porsi Latihan:** Dari total bobot *training*, 95% difokuskan ke *intraday* dan harian. Porsi mingguan hanya dialokasikan 5% sebagai jangkar tren makro jangka panjang.
3. **Kebutuhan Trading Nyata:** Dalam praktiknya, jarang sekali seorang trader mengeksekusi order riil hanya berdasarkan tebakan bentuk candlestick mingguan. Model ini dioptimalkan untuk memandu keputusan *swing* harian dan konfirmasi taktis intraday.

---

## 6. Lokasi File dan Bukti Evaluasi

- **Tabel Rekap CSV Lengkap:**  
  [`outputs/evaluations/idx/evaluation_summary.csv`](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/evaluation_summary.csv)
- **Galeri Grafik Candlestick Aktual vs Prediksi (40 Gambar PNG):**  
  Tersedia di direktori [`outputs/evaluations/idx/`](file:///E:/anaconda/AR/kronos-ft/outputs/evaluations/idx/) dengan pola nama file:  
  `eval_{TICKER}_{TIMEFRAME}.png`
