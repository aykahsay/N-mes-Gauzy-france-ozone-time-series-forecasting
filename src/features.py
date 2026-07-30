"""Feature engineering shared by the analysis notebook, training script, and dashboard."""
import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 7, 14, 21]
FOURIER_PERIOD = 365.25
FOURIER_K = 2


def fourier_terms(index: pd.DatetimeIndex, period: float = FOURIER_PERIOD, K: int = FOURIER_K) -> pd.DataFrame:
    t = np.arange(len(index))
    terms = {}
    for k in range(1, K + 1):
        terms[f"sin{k}"] = np.sin(2 * np.pi * k * t / period)
        terms[f"cos{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(terms, index=index)


def build_features(series: pd.Series, K: int = FOURIER_K, period: float = FOURIER_PERIOD) -> pd.DataFrame:
    feat = pd.DataFrame({"y": series})
    for lag in LAGS:
        feat[f"lag{lag}"] = series.shift(lag)
    feat["roll_mean_7"] = series.shift(1).rolling(7).mean()
    feat["roll_std_7"] = series.shift(1).rolling(7).std()
    feat["roll_mean_14"] = series.shift(1).rolling(14).mean()
    feat["dow"] = series.index.dayofweek
    feat["month"] = series.index.month
    t = np.arange(len(series))
    for k in range(1, K + 1):
        feat[f"sin{k}"] = np.sin(2 * np.pi * k * t / period)
        feat[f"cos{k}"] = np.cos(2 * np.pi * k * t / period)
    return feat


FEATURE_COLS = [c for c in build_features(pd.Series([0.0], index=pd.date_range("2020-01-01", periods=1))).columns if c != "y"]


def recursive_forecast(model, history: pd.Series, horizon_index: pd.DatetimeIndex) -> pd.Series:
    history = history.copy()
    preds = []
    for date in horizon_index:
        extended = pd.concat([history, pd.Series([np.nan], index=[date])])
        x_row = build_features(extended).loc[[date], FEATURE_COLS]
        pred = model.predict(x_row)[0]
        preds.append(pred)
        history.loc[date] = pred
    return pd.Series(preds, index=horizon_index)
