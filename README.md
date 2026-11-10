# USD/IDR Currency Forecasting & Volatility Platform 📈

Aplikasi prediksi dan analisis runut waktu (*time-series forecasting*) nilai tukar mata uang Rupiah terhadap Dolar AS (USD/IDR) yang membandingkan berbagai paradigma model:

1. **Model Statistik & Ekonometrika**:
   - **ARIMA (Autoregressive Integrated Moving Average)**: Univariat tren & autoregresif.
   - **ARIMAX**: Memasukkan variabel eksogen pasar global.
   - **GARCH(1,1)**: Memodelkan dinamika volatilitas (*volatility clustering*).
2. **Machine Learning Tabular (Gradient Boosting)**:
   - **XGBoost Regressor**
   - **LightGBM Regressor**
   - Dilengkapi *feature engineering*: Lagged features (H-1 s.d H-21), Moving Averages, Rolling Volatility, Log Returns, dan RSI.
3. **Deep Learning**:
   - **PyTorch LSTM (Long Short-Term Memory)** dengan sequence window multidimensi.
4. **Hybrid / Ensemble**:
   - Model agregasi rata-rata ensemble multi-paradigma.

---

## 🚀 Cara Menjalankan Aplikasi

1. Buka terminal atau PowerShell di folder proyek ini (`H:\App_Koding\Riset`).
2. Pastikan paket yang dibutuhkan sudah terpasang:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
4. Buka peramban (browser) di alamat: `http://localhost:8501`.

---

## 🛠️ Fitur Utama Aplikasi

- **Otomatisasi Data**: Pengambilan data historis USD/IDR dan faktor eksternal (Indeks Dolar DXY, Emas, Minyak Mentah WTI, US Treasury Yield 10Y, S&P 500, IHSG) langsung melalui Yahoo Finance, lengkap dengan mekanisme *caching* dan *fallback* realistis jika offline.
- **Upload CSV Kustom**: Anda dapat mengunggah file CSV dataset sendiri kapan saja.
- **Papan Peringkat Akurasi (Benchmark)**: Evaluasi metrik komparatif instan:
  - **RMSE** (Root Mean Squared Error)
  - **MAE** (Mean Absolute Error)
  - **MAPE** (Mean Absolute Percentage Error %)
  - **R² Score**
  - **Directional Accuracy (%)** (Ketepatan prediksi arah pergerakan naik/turun)
- **Visualisasi Interaktif Plotly**: Grafik perbandingan data aktual vs semua model, candlestick/garis historis, heatmap korelasi makro, dan indikator teknikal (MA50, MA200).
- **Pemodelan Volatilitas GARCH**: Mengukur lonjakan risiko dan ketidakpastian pasar.
- **Proyeksi Masa Depan (Forecasting N-Days)**: Estimasi pergerakan kurs beberapa hari ke depan lengkap dengan *95% Confidence Interval Band*.
