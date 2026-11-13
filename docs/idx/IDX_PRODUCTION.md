# Panduan Implementasi Produksi Kronos-FT IDX (Live Trading Playbook)

Panduan ini merangkum praktik terbaik saat menerapkan model fondasi finansial **Kronos-FT IDX** pada perdagangan nyata (*live trading*) di Bursa Efek Indonesia (BEI).

---

## 1. Peran Model dalam Sistem Produksi

Kronos-FT IDX tidak dirancang untuk mengambil keputusan sendirian secara membabi buta. Di dalam arsitektur qIDX v2.0, model ini berperan sebagai **filter konfirmasi arah (*directional filter*) dan penguji momentum**:

- **Fokus Utama:** Mendukung keputusan **Swing Trading Harian (`Daily` / horizon 3–5 hari bursa)** serta **eksekusi taktis intraday (`1T`–`1H`)**.
- **Kombinasi Sinyal:** Proyeksi harga model dipakai untuk memverifikasi apakah momentum ke depan searah dengan kalkulasi algoritma kuantitatif Dual-Engine qIDX sebelum order beli benar-benar dikirim ke pasar.

---

## 2. Setelan Parameter Wajib (Locked Production Specs)

Berdasarkan hasil pengujian empiris, gunakan parameter berikut untuk menjaga akurasi arah tetap tinggi ($\ge 70\%$) dan meminimalkan deviasi harga (rata-rata MAPE harian $\le 3,5\%$):

| Parameter | Nilai Produksi | Alasan Penggunaan |
| :--- | :---: | :--- |
| **`lookback_window`** | **256 bar** | Panjang memori paling ideal; menjaga korelasi arah (*IC*) tetap positif tanpa membebani model dengan data usang (*attention dilution*). |
| **`pred_len` (Daily)** | **3 – 5 hari** | Rentang waktu optimal untuk 1 putaran *swing*; mencegah penumpukan kesalahan tebakan (*compounding error*) jika rentang terlalu panjang. |
| **`pred_len` (Intraday)** | **16 bar** | Horizon yang pas untuk membaca kelanjutan tren atau pantulan *support/resistance* intraday. |
| **`temperature`** | **0.3** | Nilai rendah untuk meredam keacakan tebakan, sehingga model konsisten memilih jalur probabilitas tertinggi. |
| **`sample_count`** | **5** | Mengambil rata-rata dari 5 sampel proyeksi agar estimasi titik harga tidak mudah bergoyang oleh satu jalur acak. |

---

## 3. Disiplin Operasional: EOD Rolling Forecast

Aturan emas dalam memakai model autoregresif adalah: **jangan pernah membiarkan proyeksi basi**.

1. **Jadwal Eksekusi Harian (Pukul 16:15 WIB):**
   - Begitu sesi penutupan (*pre-closing*) bursa selesai, masukkan data harga penutupan resmi hari ini (`adj_close`, volume, dan transaksi) ke dalam DuckDB (`db/idx_market.duckdb`).
2. **Tarik 256 Bar Terakhir:**
   - Ambil 256 bar terbaru untuk emiten yang masuk radar, lalu jalankan model untuk memproyeksikan pergerakan **3–5 hari ke depan**.
3. **Syarat Konfirmasi Order Beli (*Confluence Rule*):**
   - Order beli hanya boleh dieksekusi jika kedua syarat ini terpenuhi bersamaan:
     $$\text{Arah Prediksi Kronos } (P_{t+5} > P_t) \quad \text{DAN} \quad \text{Skor Sinyal qIDX Swing} \ge 70$$
4. **Geser Jendela Setiap Sore (*Rolling Update*):**
   - Setiap hari bursa baru ditutup, jendela memori digeser satu langkah maju ($T+1$). Dengan cara ini, proyeksi selalu diperbarui berdasarkan data paling segar dan MAPE riil harian bisa terjaga stabil di kisaran **1,5% s/d 3,5%**.

---

## 4. Penyelarasan dengan Aturan Main Pasar Modal BEI

Model AI murni membaca pola deret angka dan tidak memahami regulasi bursa. Karena itu, modul eksekusi wajib mengunci batasan-batasan aturan main pasar:

1. **Pembulatan Fraksi Harga Resmi BEI:**
   - Target Profit (TP) selalu dibulatkan ke atas (*ceil tick*).
   - Stop Loss (SL) selalu dibulatkan ke bawah (*floor tick*).
2. **Batas Anti-Mengejar Harga (*Anti-Chasing Limit*):**
   - Batas harga beli pembukaan maksimal: $P_{\text{limit}} \le P_{\text{close}} + 0,5 \times \text{ATR}_{14}$. Jangan mengejar harga (*chasing*) jika saham langsung dibuka melompat jauh di atas batas ini.
3. **Filter Pengaman Dividen dan Status Saham:**
   - **Jendela Blackout Dividen:** Hindari masuk posisi beli pada rentang $[\text{CumDate}-3\text{d}, \text{ExDate}+1\text{d}]$ untuk saham dengan estimasi *dividend yield* $\ge 3,5\%$ demi menghindari jebakan penurunan harga pasca *cum-date* (*dividend trap*).
   - **Eliminasi Total:** Saham Papan Pemantauan Khusus (FCA) dan emiten yang sedang berstatus UMA aktif atau suspensi **wajib diblokir 100%**.

---

## 5. Batasan Penggunaan

> [!CAUTION]
> **Hindari penggunaan model untuk memprediksi candlestick mingguan (*weekly candlestick* / horizon > 5 hari bursa)!**  
> 1. **Pelebaran Volatilitas:** Dalam jangka 1 bulan (4 bar mingguan), ruang gerak ketidakpastian harga kumulatif melebar sekitar 4,5 kali lipat ($\sigma\sqrt{\Delta t}$). Bentuk tubuh lilin (*body*) dan ekor (*shadow*) akan sangat mudah meleset.  
> 2. **Pengaruh Fundamental Eksternal:** Tren jangka beberapa minggu ke atas sangat dipengaruhi oleh rilis laporan keuangan, pengumuman dividen, kebijakan suku bunga, atau sentimen makro yang tidak mungkin terbaca hanya dari grafik candlestick masa lalu.  
> 3. **Fokus pada Horizon Pendek:** Batasi pemanfaatan model **hanya pada rentang harian 3–5 hari bursa** dengan metode *Rolling Forecast*.
