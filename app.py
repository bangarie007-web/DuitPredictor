import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
import datetime

from src.data_loader import fetch_market_data, parse_uploaded_csv, ASSET_LABELS
from src.features import create_features, create_sequences_for_lstm
from src.models import (
    calculate_metrics,
    train_arima,
    train_arimax,
    fit_garch_volatility,
    train_xgboost,
    train_lightgbm,
    train_lstm_pytorch,
)

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="USD/IDR FX Forecasting Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">📈 USD/IDR Currency Forecasting & Volatility Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Platform Komparasi Multi-Model: Ekonometrika (ARIMA/GARCH), Machine Learning (XGBoost/LightGBM), & Deep Learning (PyTorch LSTM)</div>',
    unsafe_allow_html=True,
)

# ==========================================
# SIDEBAR: DATA SOURCE & CONFIGURATION
# ==========================================
st.sidebar.header("⚙️ Sumber Data & Konfigurasi")

data_source_opt = st.sidebar.radio(
    "Pilih Sumber Dataset:",
    ["Otomatis (Yahoo Finance & Makro)", "Upload File CSV Manual"],
)

selected_exog = ["DXY", "GOLD", "OIL", "SP500", "US10Y", "IHSG"]

if data_source_opt == "Otomatis (Yahoo Finance & Makro)":
    col_d1, col_d2 = st.sidebar.columns(2)
    start_date = col_d1.date_input("Mulai Tanggal", datetime.date(2021, 1, 1))
    end_date = col_d2.date_input("Sampai Tanggal", datetime.date.today())

    st.sidebar.markdown("**Variabel Makro Eksternal:**")
    selected_exog = st.sidebar.multiselect(
        "Pilih indikator pasar global:",
        options=list(ASSET_LABELS.keys())[1:],
        default=["DXY", "GOLD", "OIL", "US10Y"],
        format_func=lambda x: f"{x} - {ASSET_LABELS[x]}",
    )
    force_refresh = st.sidebar.checkbox("Refresh Data (Abaikan Cache)", value=False)

    with st.spinner("Mengambil dataset pasar terkini..."):
        df, data_status = fetch_market_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            selected_exog=selected_exog,
            force_refresh=force_refresh,
        )
else:
    uploaded_file = st.sidebar.file_uploader("Unggah File CSV (berisi tanggal & kurs):", type=["csv"])
    if uploaded_file is not None:
        try:
            df, data_status = parse_uploaded_csv(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file CSV: {e}")
            st.stop()
    else:
        st.info("Silakan unggah file CSV atau beralih ke mode otomatis (Yahoo Finance).")
        st.stop()

st.sidebar.info(f"📊 **Status Data:** {data_status}\n\n**Jumlah Observasi:** {len(df)} hari bursa")

# Parameter Train/Test Split
st.sidebar.header("🧪 Pembagian Data (Train/Test)")
test_size_pct = st.sidebar.slider("Proporsi Data Testing (%):", min_value=10, max_value=35, value=20, step=5)

# Pilihan Model yang akan Dijalankan
st.sidebar.header("🤖 Pilih Model untuk Dieksekusi")
run_arima = st.sidebar.checkbox("1. ARIMA (Univariat)", value=True)
run_arimax = st.sidebar.checkbox("1. ARIMAX (dengan Faktor Makro)", value=True)
run_garch = st.sidebar.checkbox("1. GARCH(1,1) Volatilitas", value=True)
run_xgb = st.sidebar.checkbox("2. XGBoost Regressor", value=True)
run_lgb = st.sidebar.checkbox("2. LightGBM Regressor", value=True)
run_lstm = st.sidebar.checkbox("3. PyTorch LSTM (Deep Learning)", value=True)
run_ensemble = st.sidebar.checkbox("4. Ensemble / Hybrid Average", value=True)

# Tombol Eksekusi
btn_train = st.sidebar.button("🚀 Latih & Bandingkan Model", type="primary", use_container_width=True)

# ==========================================
# RINGKASAN DATA HISTORIS & GRAFIK TREN
# ==========================================
tab_summary, tab_models, tab_garch, tab_future = st.tabs(
    ["📊 Data Historis & Makro", "🏆 Komparasi Prediksi Model", "⚡ Analisis Volatilitas GARCH", "🔮 Proyeksi Masa Depan"]
)

with tab_summary:
    st.subheader("Tren Historis Nilai Tukar USD/IDR & Indikator Terkait")

    # Metrics Ringkasan Terakhir
    latest_val = df["USD_IDR"].iloc[-1]
    prev_val = df["USD_IDR"].iloc[-2] if len(df) > 1 else latest_val
    diff_val = latest_val - prev_val
    diff_pct = (diff_val / prev_val) * 100

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Kurs USD/IDR Terakhir", f"Rp {latest_val:,.2f}", f"{diff_val:+,.2f} ({diff_pct:+.2f}%)")
    col_m2.metric("Nilai Tertinggi (High)", f"Rp {df['USD_IDR'].max():,.2f}")
    col_m3.metric("Nilai Terendah (Low)", f"Rp {df['USD_IDR'].min():,.2f}")
    col_m4.metric("Rata-rata Kurs", f"Rp {df['USD_IDR'].mean():,.2f}")

    # Visualisasi Kurs Interaktif
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Scatter(
            x=df.index,
            y=df["USD_IDR"],
            mode="lines",
            name="USD/IDR",
            line=dict(color="#2563EB", width=2),
        )
    )
    # Moving Average 50 & 200 untuk visualisasi tren
    ma50 = df["USD_IDR"].rolling(50).mean()
    ma200 = df["USD_IDR"].rolling(200).mean()
    fig_hist.add_trace(go.Scatter(x=df.index, y=ma50, mode="lines", name="MA 50 Hari", line=dict(dash="dot", color="#F59E0B")))
    fig_hist.add_trace(go.Scatter(x=df.index, y=ma200, mode="lines", name="MA 200 Hari", line=dict(dash="dash", color="#EF4444")))

    fig_hist.update_layout(
        title="Pergerakan Nilai Tukar USD/IDR dengan Indikator Tren",
        xaxis_title="Tanggal",
        yaxis_title="Rupiah (IDR)",
        hovermode="x unified",
        template="plotly_white",
        height=450,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Tabel Data
    with st.expander("🔍 Lihat Cuplikan Tabel Data"):
        st.dataframe(df.tail(15), use_container_width=True)

    # Korelasi antar variabel makro
    if len(df.columns) > 1:
        st.subheader("Matriks Korelasi dengan Variabel Makroekonomi")
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        corr = numeric_df.corr()
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="Viridis",
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
            )
        )
        fig_corr.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_corr, use_container_width=True)


# ==========================================
# TRAINING & EVALUASI MULTI-MODEL
# ==========================================
if btn_train or "results" in st.session_state:
    if btn_train:
        with st.spinner("Mengekstraksi fitur teknikal & melatih seluruh model..."):
            # 1. Bersihkan dataframe dasar untuk training (hindari kolom sementara)
            base_cols = [c for c in df.columns if not c.startswith("MA")]
            df_clean = df[base_cols].copy()

            # 2. Persiapan Fitur Machine Learning
            X, y, feat_cols = create_features(df_clean, target_col="USD_IDR")
            
            # Split time-series secara temporal (tanpa shuffle) berdasarkan data fitur yang valid
            n_samples = len(X)
            test_n = max(5, int(n_samples * (test_size_pct / 100)))
            split_idx = n_samples - test_n
            
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            test_dates = y_test.index

            predictions = {}
            metrics_table = []

            # 1. ARIMA Univariat (dilatih pada deret target yang sejajar)
            if run_arima:
                try:
                    p_arima, _ = train_arima(y_train, test_length=len(y_test), order=(2, 1, 2))
                    p_arima = np.asarray(p_arima, dtype=np.float64).flatten()
                    predictions["ARIMA"] = p_arima
                    m = calculate_metrics(y_test.values, p_arima)
                    metrics_table.append({"Model": "ARIMA (Univariat)", **m})
                except Exception as e:
                    st.warning(f"Gagal melatih ARIMA: {e}")

            # 1b. ARIMAX (dengan eksogen)
            if run_arimax:
                try:
                    exog_cols = [c for c in df_clean.columns if c != "USD_IDR"]
                    if exog_cols:
                        exog_aligned = df_clean.loc[y.index, exog_cols]
                        train_ex = exog_aligned.iloc[:split_idx]
                        test_ex = exog_aligned.iloc[split_idx:]
                        p_arimax, _ = train_arimax(y_train, y_test, train_ex, test_ex)
                        p_arimax = np.asarray(p_arimax, dtype=np.float64).flatten()
                        predictions["ARIMAX"] = p_arimax
                        m = calculate_metrics(y_test.values, p_arimax)
                        metrics_table.append({"Model": "ARIMAX (Makroekonomi)", **m})
                except Exception as e:
                    st.warning(f"Gagal melatih ARIMAX: {e}")

            # 2a. XGBoost
            if run_xgb:
                try:
                    p_xgb, _ = train_xgboost(X_train, y_train, X_test)
                    p_xgb = np.asarray(p_xgb, dtype=np.float64).flatten()
                    predictions["XGBoost"] = p_xgb
                    m = calculate_metrics(y_test.values, p_xgb)
                    metrics_table.append({"Model": "XGBoost", **m})
                except Exception as e:
                    st.warning(f"Gagal melatih XGBoost: {e}")

            # 2b. LightGBM
            if run_lgb:
                try:
                    p_lgb, _ = train_lightgbm(X_train, y_train, X_test)
                    p_lgb = np.asarray(p_lgb, dtype=np.float64).flatten()
                    predictions["LightGBM"] = p_lgb
                    m = calculate_metrics(y_test.values, p_lgb)
                    metrics_table.append({"Model": "LightGBM", **m})
                except Exception as e:
                    st.warning(f"Gagal melatih LightGBM: {e}")

            # 3. PyTorch LSTM
            if run_lstm:
                try:
                    # Menggunakan X (fitur yang sudah terisi dan terhindar dari NaN)
                    scaler_X = MinMaxScaler()
                    scaler_y = MinMaxScaler()
                    
                    X_scaled = scaler_X.fit_transform(X)
                    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
                    
                    seq_len = min(15, max(3, len(X_train) // 8))
                    
                    # Buat sequences langsung dari data training dan testing
                    # Untuk testing, kita sertakan seq_len data terakhir dari training agar output berukuran persis len(X_test)
                    full_scaled = np.column_stack([y_scaled, X_scaled])
                    
                    X_tr_seq, y_tr_seq = [], []
                    for i in range(len(X_train) - seq_len):
                        X_tr_seq.append(full_scaled[i : i + seq_len])
                        y_tr_seq.append(y_scaled[i + seq_len])
                    
                    X_tr_seq = np.array(X_tr_seq, dtype=np.float32)
                    y_tr_seq = np.array(y_tr_seq, dtype=np.float32)
                    
                    # Test sequence
                    test_start_idx = split_idx - seq_len
                    X_te_seq = []
                    for i in range(test_n):
                        idx = test_start_idx + i
                        X_te_seq.append(full_scaled[idx : idx + seq_len])
                    X_te_seq = np.array(X_te_seq, dtype=np.float32)

                    p_lstm_scaled, _ = train_lstm_pytorch(X_tr_seq, y_tr_seq, X_te_seq, epochs=30)
                    p_lstm = scaler_y.inverse_transform(p_lstm_scaled.reshape(-1, 1)).flatten()
                    p_lstm = np.asarray(p_lstm, dtype=np.float64)

                    predictions["LSTM (PyTorch)"] = p_lstm
                    m = calculate_metrics(y_test.values, p_lstm)
                    metrics_table.append({"Model": "PyTorch LSTM", **m})
                except Exception as e:
                    st.warning(f"Gagal melatih LSTM: {e}")

            # 4. Ensemble (Rata-rata dari seluruh model yang berhasil)
            if run_ensemble and len(predictions) >= 2:
                # Pastikan seluruh prediksi memiliki panjang yang identik dengan y_test
                aligned_preds = []
                for k, v in predictions.items():
                    v_arr = np.asarray(v, dtype=np.float64).flatten()
                    if len(v_arr) >= len(y_test):
                        aligned_preds.append(v_arr[-len(y_test):])
                
                if len(aligned_preds) >= 2:
                    ens_matrix = np.vstack(aligned_preds)  # Shape: (n_models, len(y_test))
                    ens_pred = np.mean(ens_matrix, axis=0)
                    predictions["Ensemble (Hybrid Avg)"] = ens_pred
                    m = calculate_metrics(y_test.values, ens_pred)
                    metrics_table.append({"Model": "Ensemble (Hybrid)", **m})

            # Simpan hasil di session_state
            st.session_state["results"] = {
                "predictions": predictions,
                "metrics_table": metrics_table,
                "y_test": y_test,
                "test_dates": test_dates,
            }

    # Tampilkan Hasil di Tab Model
    res = st.session_state["results"]
    y_test = res["y_test"]
    test_dates = res["test_dates"]
    predictions = res["predictions"]
    metrics_table = res["metrics_table"]

    with tab_models:
        st.subheader("🏆 Papan Peringkat Akurasi Model (Evaluation Benchmark)")
        df_metrics = pd.DataFrame(metrics_table)
        if not df_metrics.empty:
            df_metrics = df_metrics.sort_values(by="MAPE (%)", ascending=True).reset_index(drop=True)
            st.dataframe(df_metrics, use_container_width=True)

            best_model_name = df_metrics.iloc[0]["Model"]
            best_mape = df_metrics.iloc[0]["MAPE (%)"]
            st.success(f"🥇 **Model Terbaik:** **{best_model_name}** dengan MAPE terendah yaitu **{best_mape}%**.")

        # Plot Perbandingan Aktual vs Seluruh Model
        st.subheader("Grafik Perbandingan Data Aktual vs Prediksi Model pada Test Set")
        fig_comp = go.Figure()
        fig_comp.add_trace(
            go.Scatter(
                x=test_dates,
                y=y_test.values,
                mode="lines",
                name="Aktual (Ground Truth)",
                line=dict(color="#111827", width=3),
            )
        )

        colors = ["#2563EB", "#059669", "#D97706", "#DC2626", "#7C3AED", "#DB2777"]
        for i, (name, pred) in enumerate(predictions.items()):
            p_len = len(pred)
            dates_subset = test_dates[-p_len:]
            fig_comp.add_trace(
                go.Scatter(
                    x=dates_subset,
                    y=pred,
                    mode="lines",
                    name=name,
                    line=dict(dash="dash" if "ARIMA" in name else "solid", color=colors[i % len(colors)], width=2),
                )
            )

        fig_comp.update_layout(
            title="Prediksi Kurs USD/IDR vs Nilai Aktual",
            xaxis_title="Tanggal",
            yaxis_title="Kurs (IDR)",
            hovermode="x unified",
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Tab GARCH Volatilitas
    with tab_garch:
        st.subheader("⚡ Pemodelan Volatilitas Pasar dengan GARCH(1,1)")
        st.markdown(
            "Model **GARCH(1,1)** menangkap efek *volatility clustering* (periode tenang vs periode guncangan kurs) untuk memetakan risiko fluktuasi nilai tukar Rupiah."
        )
        try:
            cond_vol, garch_res = fit_garch_volatility(df["USD_IDR"])
            fig_garch = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Return Harian USD/IDR (%)", "Estimasi Volatilitas Kondisional (GARCH)"))
            returns = 100 * df["USD_IDR"].pct_change().dropna()
            
            fig_garch.add_trace(go.Scatter(x=returns.index, y=returns, mode="lines", name="Daily Return (%)", line=dict(color="#6B7280", width=1)), row=1, col=1)
            fig_garch.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol, mode="lines", name="GARCH Volatility (σ)", line=dict(color="#DC2626", width=2)), row=2, col=1)
            fig_garch.update_layout(template="plotly_white", height=550, hovermode="x unified")
            st.plotly_chart(fig_garch, use_container_width=True)

            st.markdown(f"**Ringkasan Koefisien GARCH:** Alpha = `{garch_res.params.get('alpha[1]', 0.1):.4f}`, Beta = `{garch_res.params.get('beta[1]', 0.85):.4f}`, AIC = `{garch_res.aic:.1f}`")
        except Exception as e:
            st.warning(f"Gagal menghitung GARCH: {e}")

    # Tab Proyeksi Masa Depan (Multi-Horison: Harian, Mingguan, Bulanan, Tahunan)
    with tab_future:
        st.subheader("🔮 Proyeksi Nilai Tukar USD/IDR Masa Depan")
        st.markdown(
            "Pilih horison waktu peramalan: **Harian**, **Mingguan**, **Bulanan**, atau **Tahunan** (hingga 1–2 tahun ke depan) lengkap dengan estimasi skenario (*Expected*, *Bullish/Depresiasi*, dan *Bearish/Apresiasi*)."
        )

        col_h1, col_h2 = st.columns([1, 1])
        with col_h1:
            horizon_mode = st.radio(
                "Pilih Horison Waktu Proyeksi:",
                ["📅 Harian (Daily)", "📆 Mingguan (Weekly)", "🗓️ Bulanan (Monthly)", "🪐 Tahunan (Yearly)"],
                horizontal=True,
            )

        with col_h2:
            if horizon_mode == "📅 Harian (Daily)":
                n_steps = st.slider("Jumlah Hari Kerja Kedepan:", min_value=3, max_value=60, value=14)
                freq_code = "B"
                step_label = "Hari"
            elif horizon_mode == "📆 Mingguan (Weekly)":
                n_steps = st.slider("Jumlah Minggu Kedepan:", min_value=1, max_value=52, value=12)
                freq_code = "W-FRI"
                step_label = "Minggu"
            elif horizon_mode == "🗓️ Bulanan (Monthly)":
                n_steps = st.slider("Jumlah Bulan Kedepan:", min_value=1, max_value=24, value=12)
                freq_code = "ME"
                step_label = "Bulan"
            else:
                n_steps = st.slider("Jumlah Tahun Kedepan:", min_value=1, max_value=3, value=1)
                freq_code = "YE"
                step_label = "Tahun"

        last_val = df["USD_IDR"].iloc[-1]
        last_date = df.index[-1]

        # Tentukan frekuensi dan tanggal masa depan
        if horizon_mode == "📅 Harian (Daily)":
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_steps)
            # Drift harian dari 60 hari terakhir & volatilitas harian
            recent_ret = np.log(df["USD_IDR"] / df["USD_IDR"].shift(1)).tail(60).mean()
            recent_std = np.log(df["USD_IDR"] / df["USD_IDR"].shift(1)).tail(60).std()
            step_scale = 1.0
        elif horizon_mode == "📆 Mingguan (Weekly)":
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=3), periods=n_steps, freq="W-FRI")
            # Resample weekly
            df_w = df["USD_IDR"].resample("W-FRI").last().dropna()
            recent_ret = np.log(df_w / df_w.shift(1)).tail(26).mean()
            recent_std = np.log(df_w / df_w.shift(1)).tail(26).std()
            step_scale = np.sqrt(5)
        elif horizon_mode == "🗓️ Bulanan (Monthly)":
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=10), periods=n_steps, freq="ME")
            df_m = df["USD_IDR"].resample("ME").last().dropna()
            recent_ret = np.log(df_m / df_m.shift(1)).tail(24).mean()
            recent_std = np.log(df_m / df_m.shift(1)).tail(24).std()
            step_scale = np.sqrt(21)
        else:
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=n_steps, freq="YE")
            df_y = df["USD_IDR"].resample("YE").last().dropna()
            # Gunakan drift tahunan jangka panjang historis (rata-rata depresiasi Rupiah ~2-4% per tahun)
            if len(df_y) >= 2:
                recent_ret = np.log(df_y / df_y.shift(1)).tail(5).mean()
                recent_std = np.log(df_y / df_y.shift(1)).tail(5).std()
            else:
                recent_ret = 0.03  # Default 3% drift tahunan jika data tahunan pendek
                recent_std = 0.05
            step_scale = np.sqrt(252)

        if np.isnan(recent_ret):
            recent_ret = 0.0001
        if np.isnan(recent_std) or recent_std == 0:
            recent_std = 0.005

        # Hitung Proyeksi Roll-Forward & Skenario
        future_proj = []
        upper_band = []
        lower_band = []
        curr = last_val

        for i in range(1, n_steps + 1):
            curr = curr * np.exp(recent_ret)
            future_proj.append(curr)
            # Confidence Interval 95% dengan penyebaran akar waktu (sqrt of time)
            band = curr * (1.96 * recent_std * np.sqrt(i))
            upper_band.append(curr + band)
            lower_band.append(curr - band)

        # Ringkasan KPI Proyeksi Akhir
        end_proj = future_proj[-1]
        delta_proj = end_proj - last_val
        delta_pct = (delta_proj / last_val) * 100

        kp1, kp2, kp3, kp4 = st.columns(4)
        kp1.metric("Kurs Saat Ini", f"Rp {last_val:,.2f}")
        kp2.metric(f"Ekspektasi ({n_steps} {step_label})", f"Rp {end_proj:,.2f}", f"{delta_proj:+,.2f} ({delta_pct:+.2f}%)")
        kp3.metric("Skenario Terlemah (Batas Atas)", f"Rp {upper_band[-1]:,.2f}")
        kp4.metric("Skenario Terkuat (Batas Bawah)", f"Rp {lower_band[-1]:,.2f}")

        # Dataframe Proyeksi
        date_format = "%Y-%m-%d" if horizon_mode in ["📅 Harian (Daily)", "📆 Mingguan (Weekly)"] else ("%Y-%m" if horizon_mode == "🗓️ Bulanan (Monthly)" else "%Y")
        df_future = pd.DataFrame(
            {
                "Periode Target": future_dates.strftime(date_format),
                "Estimasi Kurs (Rp)": np.round(future_proj, 2),
                "Batas Bawah (Apresiasi IDR)": np.round(lower_band, 2),
                "Batas Atas (Depresiasi IDR)": np.round(upper_band, 2),
            }
        )

        st.dataframe(df_future, use_container_width=True)

        # Plot Proyeksi Interaktif
        fig_fut = go.Figure()
        # Ambil konteks historis yang proporsional
        ctx_len = min(60, len(df)) if horizon_mode == "📅 Harian (Daily)" else min(120, len(df))
        context_df = df["USD_IDR"].tail(ctx_len)

        fig_fut.add_trace(go.Scatter(x=context_df.index, y=context_df.values, mode="lines", name="Historis Aktual", line=dict(color="#1E3A8A", width=2)))
        fig_fut.add_trace(go.Scatter(x=future_dates, y=future_proj, mode="lines+markers", name=f"Proyeksi ({horizon_mode})", line=dict(color="#10B981", width=2.5)))
        fig_fut.add_trace(go.Scatter(x=future_dates, y=upper_band, mode="lines", name="Skenario Risiko 95% (Upper)", line=dict(color="rgba(239, 68, 68, 0.3)", width=1, dash="dot")))
        fig_fut.add_trace(go.Scatter(x=future_dates, y=lower_band, mode="lines", fill="tonexty", fillcolor="rgba(16, 185, 129, 0.12)", name="Rentang Ketidakpastian 95%", line=dict(color="rgba(16, 185, 129, 0.3)", width=1, dash="dot")))

        fig_fut.update_layout(
            title=f"Grafik Proyeksi Nilai Tukar USD/IDR - Horison {horizon_mode}",
            xaxis_title="Periode Waktu",
            yaxis_title="Kurs (IDR)",
            template="plotly_white",
            hovermode="x unified",
            height=480,
        )
        st.plotly_chart(fig_fut, use_container_width=True)

else:
    with tab_models:
        st.info("👈 Silakan pilih model yang diinginkan pada panel sidebar di sebelah kiri, lalu klik tombol **'Latih & Bandingkan Model'**.")
    with tab_garch:
        st.info("Klik tombol **'Latih & Bandingkan Model'** di sidebar untuk menjalankan simulasi GARCH.")
    with tab_future:
        st.info("Klik tombol **'Latih & Bandingkan Model'** di sidebar untuk melihat proyeksi masa depan.")
