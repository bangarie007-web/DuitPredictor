import numpy as np
import pandas as pd


def create_features(
    df: pd.DataFrame,
    target_col: str = "USD_IDR",
    lags: list[int] = [1, 2, 3, 5, 7, 14, 21],
    rolling_windows: list[int] = [5, 14, 30],
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Menghasilkan fitur teknikal, lagged features, return harian, moving averages,
    volatilitas, serta momentum RSI untuk model Machine Learning tabular (XGBoost / LightGBM)
    dan Deep Learning.
    """
    data = df.copy()
    if target_col not in data.columns:
        raise ValueError(f"Kolom target {target_col} tidak ditemukan dalam DataFrame.")

    feature_cols = []

    # 1. Fitur Kalender / Siklus Waktu
    if isinstance(data.index, pd.DatetimeIndex):
        data["dayofweek"] = data.index.dayofweek
        data["day"] = data.index.day
        data["month"] = data.index.month
        data["is_month_end"] = data.index.is_month_end.astype(int)
        feature_cols.extend(["dayofweek", "day", "month", "is_month_end"])

    # 2. Lagged Target
    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        data[col_name] = data[target_col].shift(lag)
        feature_cols.append(col_name)

    # 3. Target Rolling Mean & Rolling Std (Volatility)
    for w in rolling_windows:
        rm_col = f"{target_col}_roll_mean_{w}"
        rs_col = f"{target_col}_roll_std_{w}"
        data[rm_col] = data[target_col].shift(1).rolling(window=w).mean()
        data[rs_col] = data[target_col].shift(1).rolling(window=w).std()
        feature_cols.extend([rm_col, rs_col])

    # 4. Momentum & Return Logaritmik Target
    data[f"{target_col}_ret1"] = np.log(data[target_col] / data[target_col].shift(1))
    data[f"{target_col}_ret5"] = np.log(data[target_col] / data[target_col].shift(5))
    feature_cols.extend([f"{target_col}_ret1", f"{target_col}_ret5"])

    # 5. Relative Strength Index (RSI - 14) Sederhana
    delta = data[target_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data["rsi_14"] = 100 - (100 / (1 + rs))
    feature_cols.append("rsi_14")

    # 6. Fitur dari Variabel Eksternal (Exogenous: DXY, Gold, Oil, dll)
    exog_cols = [c for c in data.columns if c != target_col and c not in feature_cols]
    for ex in exog_cols:
        # Lag 1 & Return 1 untuk exog untuk mencegah data leakage
        data[f"{ex}_lag1"] = data[ex].shift(1)
        data[f"{ex}_ret1"] = data[ex].pct_change(1).shift(1)
        feature_cols.extend([f"{ex}_lag1", f"{ex}_ret1"])

    # Drop baris NaN akibat shifting / rolling
    data = data.dropna()
    X = data[feature_cols]
    y = data[target_col]

    return X, y, feature_cols


def create_sequences_for_lstm(
    data_scaled: np.ndarray,
    seq_length: int = 30,
    forecast_horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mengubah array multidimensi terstandarisasi menjadi sliding window (sekuens)
    berukuran (samples, seq_length, n_features) untuk LSTM / Neural Network.
    Target y adalah nilai pada index ke-0 (kolom pertama diasumsikan USD_IDR).
    """
    X, y = [], []
    for i in range(len(data_scaled) - seq_length - forecast_horizon + 1):
        X.append(data_scaled[i : i + seq_length])
        # Prediksi target di horizon tertentu (misal t + 1)
        y.append(data_scaled[i + seq_length + forecast_horizon - 1, 0])
    return np.array(X), np.array(y)
