"""Shared data loading/cleaning logic used by both the analysis notebook and the dashboard."""
import pandas as pd

LONG_GAP_THRESHOLD_DAYS = 14
SERIES_START = "2017-01-01"


def load_hourly(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates(subset="Date").set_index("Date")
    return df


def to_daily(hourly_ozone: pd.Series) -> pd.Series:
    daily = hourly_ozone.resample("D").mean().asfreq("D")
    return daily.loc[SERIES_START:]


def long_gap_mask(daily_raw: pd.Series, threshold: int = LONG_GAP_THRESHOLD_DAYS) -> pd.Series:
    is_null = daily_raw.isna()
    gap_id = (is_null != is_null.shift()).cumsum()
    mask = pd.Series(False, index=daily_raw.index)
    for _, grp in daily_raw[is_null].groupby(gap_id[is_null]):
        if len(grp) > threshold:
            mask.loc[grp.index] = True
    return mask


def clean_daily(csv_path):
    """Returns (daily filled series, long_gap_mask) starting 2017-01-01."""
    hourly = load_hourly(csv_path)
    daily_raw = to_daily(hourly["Ozone"])
    mask = long_gap_mask(daily_raw)
    daily = daily_raw.interpolate(method="time").ffill().bfill()
    return daily, mask


def chronological_split(daily: pd.Series, val_frac=0.2, test_frac=0.2):
    n = len(daily)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    train = daily.iloc[: n - n_val - n_test]
    val = daily.iloc[n - n_val - n_test : n - n_test]
    test = daily.iloc[n - n_test :]
    return train, val, test
