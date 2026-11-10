import os
import datetime
import numpy as np
import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


TICKER_MAP = {
    "USD_IDR": "IDR=X",
    "DXY": "DX-Y.NYB",
    "GOLD": "GC=F",
    "OIL": "CL=F",
    "SP500": "^GSPC",
    "US10Y": "^TNX",
    "IHSG": "^JKSE",
}

ASSET_LABELS = {
    "USD_IDR": "Kurs USD/IDR (Rp)",
    "DXY": "Indeks Dolar AS (DXY)",
    "GOLD": "Harga Emas Dunia (USD/oz)",
    "OIL": "Minyak Mentah WTI (USD/bbl)",
    "SP500": "Indeks S&P 500",
    "US10Y": "US Treasury 10Y Yield (%)",
    "IHSG": "IHSG / JKSE (Indonesia)",
}


def generate_realistic_synthetic_data(
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Menghasilkan dataset makroekonomi & nilai tukar USD/IDR realistis (GARCH + cointegration drift)
    sebagai cadangan (fallback) apabila koneksi internet ke Yahoo Finance terputus atau dibatasi.
    """
    rng = np.random.default_rng(seed)
    if end_date is None:
        end_date = datetime.date.today().strftime("%Y-%m-%d")

    dates = pd.bdate_range(start=start_date, end=end_date)
    n = len(dates)
    if n < 60:
        dates = pd.bdate_range(end=end_date, periods=500)
        n = len(dates)

    # 1. Simulasikan DXY (sekitar 95 - 108)
    dxy_shocks = rng.normal(0, 0.35, size=n)
    dxy = 98.0 + np.cumsum(dxy_shocks)
    dxy = np.clip(dxy, 89.0, 114.0)

    # 2. Simulasikan US10Y Yield (1.5% - 4.8%)
    us10y_shocks = rng.normal(0.001, 0.04, size=n)
    us10y = 2.0 + np.cumsum(us10y_shocks)
    us10y = np.clip(us10y, 0.6, 5.2)

    # 3. Simulasikan Gold (1700 - 2700)
    gold_shocks = rng.normal(0.35, 14.0, size=n) - 4.0 * dxy_shocks
    gold = 1800.0 + np.cumsum(gold_shocks)
    gold = np.clip(gold, 1500.0, 3200.0)

    # 4. Simulasikan Oil (55 - 105)
    oil_shocks = rng.normal(0.01, 1.3, size=n)
    oil = 68.0 + np.cumsum(oil_shocks)
    oil = np.clip(oil, 35.0, 125.0)

    # 5. Simulasikan S&P 500 (3500 - 6000)
    sp500_ret = rng.normal(0.0004, 0.01, size=n)
    sp500 = 3800.0 * np.exp(np.cumsum(sp500_ret))

    # 6. Simulasikan IHSG (6000 - 7800)
    ihsg_ret = 0.4 * sp500_ret + rng.normal(0.00025, 0.0075, size=n)
    ihsg = 6100.0 * np.exp(np.cumsum(ihsg_ret))

    # 7. Simulasikan USD/IDR dengan GARCH volatility clustering + korelasi positif terhadap DXY & US10Y
    vol = np.zeros(n)
    eps = np.zeros(n)
    vol[0] = 0.0032
    omega = 1.5e-6
    alpha = 0.12
    beta = 0.84

    for t in range(1, n):
        vol[t] = np.sqrt(omega + alpha * (eps[t - 1] ** 2) + beta * (vol[t - 1] ** 2))
        z = rng.standard_normal()
        eps[t] = vol[t] * z

    # Pengaruh faktor eksternal terhadap return harian USD/IDR
    dxy_ret = np.diff(np.log(dxy), prepend=np.log(dxy[0]))
    us10y_diff = np.diff(us10y, prepend=us10y[0])
    ihsg_r = np.diff(np.log(ihsg), prepend=np.log(ihsg[0]))

    idr_ret = (
        0.00008  # drift depresiasi jangka panjang moderat
        + 0.42 * dxy_ret
        + 0.008 * us10y_diff
        - 0.15 * ihsg_r
        + eps
    )
    usd_idr = 14250.0 * np.exp(np.cumsum(idr_ret))
    # Tambahkan komponen musiman kuartalan/tahunan halus
    t_idx = np.arange(n)
    seasonal = 120.0 * np.sin(2 * np.pi * t_idx / 252.0) + 55.0 * np.cos(2 * np.pi * t_idx / 63.0)
    usd_idr = np.round(usd_idr + seasonal, 2)

    df = pd.DataFrame(
        {
            "USD_IDR": usd_idr,
            "DXY": np.round(dxy, 2),
            "GOLD": np.round(gold, 2),
            "OIL": np.round(oil, 2),
            "SP500": np.round(sp500, 2),
            "US10Y": np.round(us10y, 3),
            "IHSG": np.round(ihsg, 2),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def fetch_market_data(
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    selected_exog: list[str] | None = None,
    cache_dir: str = "data",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
    """
    Mengambil data historis USD/IDR dan variabel makro/pasar global dari Yahoo Finance.
    Mengembalikan (DataFrame, status_source_message).
    """
    os.makedirs(cache_dir, exist_ok=True)
    if end_date is None:
        end_date = datetime.date.today().strftime("%Y-%m-%d")

    if selected_exog is None:
        selected_exog = ["DXY", "GOLD", "OIL", "SP500", "US10Y", "IHSG"]

    cache_file = os.path.join(cache_dir, f"usdidr_market_{start_date}_{end_date}.csv")

    if os.path.exists(cache_file) and not force_refresh:
        try:
            df_cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            keep_cols = ["USD_IDR"] + [c for c in selected_exog if c in df_cached.columns]
            return df_cached[keep_cols].dropna(how="all"), "Cache Lokal (Yahoo Finance)"
        except Exception:
            pass

    if HAS_YFINANCE:
        try:
            cols_to_fetch = ["USD_IDR"] + [c for c in selected_exog if c in TICKER_MAP]
            series_dict = {}

            for col in cols_to_fetch:
                ticker = TICKER_MAP[col]
                raw = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                )
                if raw is not None and not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        close_s = raw["Close"].iloc[:, 0]
                    else:
                        close_s = raw["Close"]
                    close_s.index = pd.to_datetime(close_s.index).tz_localize(None)
                    series_dict[col] = close_s

            if "USD_IDR" in series_dict and len(series_dict["USD_IDR"]) >= 60:
                df = pd.DataFrame(series_dict)
                df.index.name = "Date"
                df = df.sort_index()
                # Filter outlier ekstrem / error tick pada IDR=X (mis. nilai < 5000 atau > 30000)
                df.loc[(df["USD_IDR"] < 5000) | (df["USD_IDR"] > 30000), "USD_IDR"] = np.nan
                df = df.ffill().bfill()
                df = df.dropna(subset=["USD_IDR"])

                try:
                    df.to_csv(cache_file)
                except Exception:
                    pass

                return df, "Live Yahoo Finance (yfinance)"
        except Exception as e:
            fallback_msg = f"Simulasi Makroekonomi Realistis (Fallback karena jaringan/yfinance: {e})"
            df_syn = generate_realistic_synthetic_data(start_date, end_date)
            keep_cols = ["USD_IDR"] + [c for c in selected_exog if c in df_syn.columns]
            return df_syn[keep_cols], fallback_msg

    df_syn = generate_realistic_synthetic_data(start_date, end_date)
    keep_cols = ["USD_IDR"] + [c for c in selected_exog if c in df_syn.columns]
    return df_syn[keep_cols], "Dataset Makroekonomi Terintegrasi (Mode Offline/Simulasi)"


def parse_uploaded_csv(uploaded_file) -> tuple[pd.DataFrame, str]:
    """
    Memproses file CSV yang diunggah pengguna.
    Mendukung deteksi otomatis kolom Tanggal dan kolom Kurs USD/IDR.
    """
    df = pd.read_csv(uploaded_file)
    if df.empty:
        raise ValueError("File CSV kosong.")

    # Cari kolom tanggal
    date_col = None
    for col in df.columns:
        if any(k in col.lower() for k in ["date", "tanggal", "waktu", "time", "tgl"]):
            date_col = col
            break
    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "Date"

    # Cari kolom target USD_IDR
    target_col = None
    for col in df.columns:
        c_low = col.lower()
        if any(k in c_low for k in ["usd_idr", "usdidr", "idr", "kurs", "close", "nilai_tukar", "rate"]):
            target_col = col
            break

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col is None:
        if not numeric_cols:
            # Coba konversi string ke float
            for c in df.columns:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("Tidak ditemukan kolom numerik pada file CSV.")
        target_col = numeric_cols[0]

    if target_col != "USD_IDR":
        df = df.rename(columns={target_col: "USD_IDR"})

    # Pastikan semua kolom lain numerik
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(axis=1, how="all").ffill().bfill()
    df = df.dropna(subset=["USD_IDR"])
    return df, f"CSV Kustom ({uploaded_file.name})"
