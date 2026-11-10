import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Menghitung metrik performa prediksi:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - MAPE (Mean Absolute Percentage Error)
    - R2 Score (Koefisien Determinasi)
    - Directional Accuracy (% ketepatan prediksi arah naik/turun)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"RMSE": 0.0, "MAE": 0.0, "MAPE (%)": 0.0, "R2": 0.0, "Dir. Accuracy (%)": 0.0}

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
    r2 = float(r2_score(y_true, y_pred))

    # Directional Accuracy (Direction of change)
    if len(y_true) > 1:
        dir_true = np.sign(np.diff(y_true))
        dir_pred = np.sign(np.diff(y_pred))
        dir_acc = float(np.mean(dir_true == dir_pred) * 100)
    else:
        dir_acc = 100.0

    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE (%)": round(mape, 2),
        "R2": round(r2, 4),
        "Dir. Accuracy (%)": round(dir_acc, 2),
    }


# ==========================================
# 1. MODEL STATISTIK & EKONOMETRIKA
# ==========================================
def train_arima(
    train_series: pd.Series | np.ndarray,
    test_length: int,
    order: tuple[int, int, int] = (1, 1, 1),
) -> tuple[np.ndarray, object]:
    """
    Model ARIMA univariat untuk menangkap tren dan auto-regresif kurs.
    Menggunakan array nilai murni untuk menghindari exception 'No supported index' pada tanggal bursa yang non-kontinu.
    """
    from statsmodels.tsa.arima.model import ARIMA

    train_arr = np.asarray(train_series, dtype=np.float64).flatten()
    model = ARIMA(train_arr, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=test_length)
    return np.asarray(forecast, dtype=np.float64).flatten(), model_fit


def train_arimax(
    train_y: pd.Series | np.ndarray,
    test_y: pd.Series | np.ndarray,
    train_exog: pd.DataFrame | np.ndarray,
    test_exog: pd.DataFrame | np.ndarray,
    order: tuple[int, int, int] = (1, 1, 1),
) -> tuple[np.ndarray, object]:
    """
    ARIMAX: ARIMA dengan variabel eksogen (makroekonomi seperti DXY, US10Y, dll.)
    Menggunakan nilai array numerik murni untuk kestabilan indeks statsmodels.
    """
    from statsmodels.tsa.arima.model import ARIMA

    y_tr_arr = np.asarray(train_y, dtype=np.float64).flatten()
    ex_tr_arr = np.asarray(train_exog, dtype=np.float64)
    ex_te_arr = np.asarray(test_exog, dtype=np.float64)
    steps = len(test_y)

    model = ARIMA(y_tr_arr, order=order, exog=ex_tr_arr)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps, exog=ex_te_arr)
    return np.asarray(forecast, dtype=np.float64).flatten(), model_fit


def fit_garch_volatility(series: pd.Series, p: int = 1, q: int = 1) -> tuple[np.ndarray, object]:
    """
    GARCH(p, q) untuk memodelkan volatilitas return kurs USD/IDR (clustering volatility).
    Mengembalikan volatilitas kondisional bersyarat.
    """
    from arch import arch_model

    # GARCH memerlukan data stationer (return persentase)
    returns = 100 * series.pct_change().dropna()
    am = arch_model(returns, vol="Garch", p=p, q=q, mean="Constant")
    res = am.fit(disp="off")
    cond_vol = res.conditional_volatility
    return cond_vol, res


# ==========================================
# 2. MODEL MACHINE LEARNING TABULAR
# ==========================================
def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_estimators: int = 250,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> tuple[np.ndarray, object]:
    """
    XGBoost Regressor untuk prediksi nilai tukar dengan ensemble tree.
    """
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_estimators: int = 250,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> tuple[np.ndarray, object]:
    """
    LightGBM Regressor (cepat, efisien dalam menangkap relasi non-linier).
    """
    import lightgbm as lgb

    # Konversi ke float numpy array 2 dimensi untuk kompatibilitas penuh antar versi LightGBM
    X_tr_arr = np.asarray(X_train, dtype=np.float64)
    y_tr_arr = np.asarray(y_train, dtype=np.float64).flatten()
    X_te_arr = np.asarray(X_test, dtype=np.float64)

    if X_tr_arr.ndim != 2 or X_tr_arr.shape[0] == 0 or X_tr_arr.shape[1] == 0:
        raise ValueError(f"Dimensi fitur tidak valid untuk LightGBM: {X_tr_arr.shape}")

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(X_tr_arr, y_tr_arr)
    preds = model.predict(X_te_arr)
    return np.asarray(preds, dtype=np.float64).flatten(), model


# ==========================================
# 3. DEEP LEARNING (PYTORCH LSTM)
# ==========================================
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class PyTorchLSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def train_lstm_pytorch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 0.003,
    hidden_dim: int = 64,
    num_layers: int = 2,
    device: str = "cpu",
) -> tuple[np.ndarray, object]:
    """
    Melatih model LSTM PyTorch dengan mini-batch gradient descent dan Adam optimizer.
    """
    torch.manual_seed(42)
    input_dim = X_train.shape[2]
    model = PyTorchLSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_in = torch.FloatTensor(X_test).to(device)
        preds = model(test_in).cpu().numpy().flatten()

    return preds, model
