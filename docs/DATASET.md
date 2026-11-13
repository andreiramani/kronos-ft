# Spesifikasi dan Cakupan Dataset (48,29 Juta Bar K-Line)

Catatan teknis ini merangkum cakupan data historis, rentang kalender, dan karakteristik mikrostruktur dataset yang digunakan untuk melatih (*fine-tuning*) model fondasi finansial **Kronos-FT** di empat bursa saham dunia: **IDX** (Indonesia), **NASDAQ** (Amerika Serikat), **TSE** (Jepang), dan **NSE** (India).

---

## 1. Ringkasan Portofolio Data

Seluruh data disusun menggunakan prinsip pemisahan kronologis (*temporal splitting*: 80% data latih, 10% validasi, 10% uji) untuk memastikan evaluasi berjalan murni tanpa kebocoran data masa depan (*zero-lookahead bias*):

- **Total Volume:** **48.294.489 bar K-Line**
- **Rentang Kalender:** 1999 sampai 2026 (kedalaman historis hingga 25+ tahun)
- **Multi-Resolusi (7 Timeframe per Bursa):** `1T` (1 menit), `5T` (5 menit), `15T`, `30T`, `1H` (1 jam), `D` (harian), dan `W` (mingguan)
- **Fitur per Bar:** `timestamps`, `open`, `high`, `low`, `close`, `volume`, dan `amount`

---

## 2. Rincian Durasi dan Kedalaman Bar per Bursa

Tabel berikut menunjukkan rentang waktu kalender serta total bar K-line yang sudah divalidasi dan siap dipakai pada masing-masing bursa:

| Bursa Saham | Timeframe | Rentang Kalender (Mulai s/d Selesai) | Total Bar (Selected Tickers) |
| :--- | :--- | :---: | :---: |
| **IDX (Indonesia)** 🇮🇩 | `1T` (1 Menit) | 1 Bulan (06 Okt 2026 – 10 Nov 2026) | 4.887.899 bars |
| | `5T` (5 Menit) | 3+ Tahun (02 Jan 2023 – 10 Nov 2026) | 2.752.946 bars |
| | `15T` (15 Menit) | 3+ Tahun (02 Jan 2023 – 10 Nov 2026) | 1.050.414 bars |
| | `30T` (30 Menit) | 3+ Tahun (02 Jan 2023 – 10 Nov 2026) | 554.435 bars |
| | `1H` (1 Jam) | 3+ Tahun (02 Jan 2023 – 10 Nov 2026) | 310.131 bars |
| | `D` (Harian) | 10+ Tahun (05 Jan 2015 – 10 Nov 2026) | 81.175 bars |
| | `W` (Mingguan) | 10+ Tahun (11 Jan 2015 – 15 Nov 2026) | 15.648 bars |
| | **Subtotal IDX** | **7 Timeframe** | **9.652.648 bars** |
| | | | |
| **NASDAQ (US)** 🇺🇸 | `1T` (1 Menit) | 1 Bulan (01 Okt 2026 – 11 Nov 2026) | 5.250.709 bars |
| | `5T` (5 Menit) | 3+ Tahun (03 Jan 2023 – 11 Nov 2026) | 2.945.188 bars |
| | `15T` (15 Menit) | 3+ Tahun (03 Jan 2023 – 11 Nov 2026) | 1.132.840 bars |
| | `30T` (30 Menit) | 3+ Tahun (03 Jan 2023 – 11 Nov 2026) | 586.294 bars |
| | `1H` (1 Jam) | 3+ Tahun (03 Jan 2023 – 11 Nov 2026) | 318.552 bars |
| | `D` (Harian) | 25+ Tahun (03 Jan 2000 – 11 Nov 2026) | 185.340 bars |
| | `W` (Mingguan) | 25+ Tahun (09 Jan 2000 – 15 Nov 2026) | 39.420 bars |
| | **Subtotal NASDAQ** | **7 Timeframe** | **10.458.343 bars** |
| | | | |
| **TSE (Jepang)** 🇯🇵 | `1T` (1 Menit) | 1 Bulan (01 Okt 2026 – 12 Nov 2026) | 3.486.220 bars |
| | `5T` (5 Menit) | 3+ Tahun (04 Jan 2023 – 12 Nov 2026) | 1.890.315 bars |
| | `15T` (15 Menit) | 3+ Tahun (04 Jan 2023 – 12 Nov 2026) | 711.450 bars |
| | `30T` (30 Menit) | 3+ Tahun (04 Jan 2023 – 12 Nov 2026) | 370.820 bars |
| | `1H` (1 Jam) | 3+ Tahun (04 Jan 2023 – 12 Nov 2026) | 198.412 bars |
| | `D` (Harian) | 7.5+ Tahun (01 Jan 2019 – 12 Nov 2026) | 24.380 bars |
| | `W` (Mingguan) | 7.5+ Tahun (06 Jan 2019 – 15 Nov 2026) | 2.600 bars |
| | **Subtotal TSE** | **7 Timeframe** | **6.684.197 bars** |
| | | | |
| **NSE (India)** 🇮🇳 | `1T` (1 Menit) | 3 Bulan (03 Agu 2026 – 12 Nov 2026) | 14.821.405 bars |
| | `5T` (5 Menit) | 7.5+ Tahun (02 Feb 2015 – 12 Nov 2026) | 3.655.782 bars |
| | `15T` (15 Menit) | 7.5+ Tahun (02 Feb 2015 – 12 Nov 2026) | 1.050.401 bars |
| | `30T` (30 Menit) | 7.5+ Tahun (02 Feb 2015 – 12 Nov 2026) | 562.165 bars |
| | `1H` (1 Jam) | 7.5+ Tahun (02 Feb 2015 – 12 Nov 2026) | 302.860 bars |
| | `D` (Harian) | 15 Tahun (14 Nov 2011 – 12 Nov 2026) | 88.798 bars |
| | `W` (Mingguan) | 15 Tahun (18 Nov 2011 – 13 Nov 2026) | 18.792 bars |
| | **Subtotal NSE** | **7 Timeframe** | **21.499.301 bars** |
| | | | |
| **TOTAL KESELURUHAN** | **7 Timeframe** | **Lintas 4 Bursa (1999 s/d 2026)** | **48.294.489 bars** |

---

## 3. Karakteristik Pasar dan Strategi Sampling

Tiap bursa memiliki karakter mikrostruktur yang berbeda, sehingga pendekatan *sampling* saat proses *training* juga disesuaikan dengan tujuan operasional masing-masing:

1. **Bursa Efek Indonesia (IDX) 🇮🇩**
   - **Universe:** 50 saham paling likuid yang mewakili 11 sektor IDX-IC.
   - **Sumber Data:** Basis data analitik DuckDB (`db/idx_market.duckdb`) dipadukan dengan *feed intraday* Stockbit.
   - **Distribusi Sampling:** Sengaja dialokasikan berbobot tinggi pada **Harian / Daily (40%)**, lalu 55% disebar ke *intraday* (`1T`–`1H`), dan 5% untuk mingguan (`W`). Fokus utamanya memang ditujukan untuk strategi *swing trading* harian di pasar domestik, sehingga model perlu melihat variasi pola *candlestick* harian lebih banyak.

2. **NASDAQ (Amerika Serikat) 🇺🇸**
   - **Universe:** 30 saham *large-cap tech & growth leaders* .
   - **Distribusi Sampling:** Menggunakan bobot seragam (*uniform* ~14,3% per *timeframe*). Likuiditas saham-saham ini sangat tebal di semua skala waktu, jadi distribusi alami sudah cukup representatif.

3. **Tokyo Stock Exchange (TSE) 🇯🇵**
   - **Universe:** 30 saham *blue-chip* konstituen Nikkei 225.
   - **Distribusi Sampling:** Bobot merata (*uniform* ~14,3% per *timeframe*) dengan memperhatikan jeda sesi perdagangan pagi dan siang khas bursa Tokyo.

4. **National Stock Exchange of India (NSE) 🇮🇳**
   - **Universe:** 30 saham teratas konstituen Nifty 50.
   - **Karakteristik Khusus:** Menjadi kontributor volume terbesar dalam dataset (21,49 juta bar), terutama karena ketersediaan data *intraday* 5-menit yang sangat panjang sejak tahun 2015.

---

