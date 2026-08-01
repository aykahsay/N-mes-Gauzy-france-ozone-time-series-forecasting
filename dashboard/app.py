"""
Streamlit dashboard for the Paris Ozone Forecasting Project.

Displays exact plots generated in the Jupyter notebook (`paris-ozone-time-series-forecasting.ipynb`)
along with summary metrics, statistical tests, and interactive forecast exploration.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import fourier_terms, recursive_forecast  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "dashboard" / "figures"

st.set_page_config(page_title="Paris Ozone Forecasting", page_icon="🌫️", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar & Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { min-width: 300px; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.95rem; opacity: 0.85; }
    .stCaption { font-size: 0.95rem !important; opacity: 0.9 !important; }
    h1 { font-size: 2.1rem !important; }
    h2, .stSubheader { font-size: 1.4rem !important; margin-top: 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_daily():
    df = pd.read_csv(PROCESSED_DIR / "daily.csv", parse_dates=["Date"], index_col="Date")
    return df["Ozone"], df["long_gap"].astype(bool)


@st.cache_data
def load_predictions():
    return pd.read_csv(PROCESSED_DIR / "predictions.csv", parse_dates=["Date"])


@st.cache_data
def load_metrics():
    with open(MODELS_DIR / "metrics.json") as f:
        return json.load(f)


daily, gap_mask = load_daily()
predictions = load_predictions()
metrics = load_metrics()

# Navigation
with st.sidebar:
    st.markdown("## 🌫️ Ozone Forecasting")
    st.caption("Paris ground-level O₃ (STA4030 Term Paper)")
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "Overview & Data Cleaning",
            "Decomposition & Stationarity",
            "Model Diagnostics (SARIMA & XGBoost)",
            "Model Comparison",
            "Forecast Explorer",
        ],
    )
    st.divider()
    st.caption(
        f"**Data Range**: {daily.index.min().date()} → {daily.index.max().date()}\n"
        f"**Total Days**: {len(daily)}\n"
        f"**Interpolated Days**: {gap_mask.sum()} ({gap_mask.mean():.0%})"
    )

st.title("Paris Ozone (O₃) Forecasting Dashboard")
st.caption(
    f"Visualizing notebook analysis & results for daily mean ground-level ozone in Paris "
    f"({daily.index.min().date()} to {daily.index.max().date()})."
)

# ---------------------------------------------------------------------------
# Page 1: Overview & Data Cleaning
# ---------------------------------------------------------------------------
if page == "Overview & Data Cleaning":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Ozone (µg/m³)", f"{daily.mean():.1f}")
    col2.metric("Max Ozone (µg/m³)", f"{daily.max():.1f}")
    col3.metric("Calendar Days", f"{len(daily)}")
    col4.metric("Interpolated Days", f"{gap_mask.sum()} ({gap_mask.mean():.0%})")

    st.subheader("1. Daily Mean Ozone Concentration (Notebook Plot)")
    img_eda = FIGURES_DIR / "22_1_4_handling_missing_data_resampling.png"
    if img_eda.exists():
        st.image(str(img_eda), use_container_width=True)
        st.caption("Figure from Notebook Cell 22: Daily mean ozone series showing interpolated gap periods.")

    st.subheader("2. Exploratory Data Analysis (Notebook Plot)")
    img_raw = FIGURES_DIR / "17_1_3_exploratory_data_analysis.png"
    if img_raw.exists():
        st.image(str(img_raw), use_container_width=True)
        st.caption("Figure from Notebook Cell 17: Exploratory distribution and raw time series.")

    st.subheader("Summary Statistics")
    st.dataframe(daily.describe().to_frame("Ozone (µg/m³)").T, use_container_width=True)

# ---------------------------------------------------------------------------
# Page 2: Decomposition & Stationarity
# ---------------------------------------------------------------------------
elif page == "Decomposition & Stationarity":
    st.subheader("1. STL Decomposition (Notebook Plot)")
    img_stl = FIGURES_DIR / "24_1_5_seasonal_trend_decomposition_stl.png"
    if img_stl.exists():
        st.image(str(img_stl), use_container_width=True)
        st.caption(
            "Figure from Notebook Cell 24: STL Decomposition with 365-day annual period (Observed, Trend, Seasonal, Residual)."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("2. Stationarity Analysis")
        from statsmodels.tsa.stattools import adfuller, kpss

        def stationarity_row(series, label):
            adf_stat, adf_p, *_ = adfuller(series.dropna(), autolag="AIC")
            kpss_stat, kpss_p, *_ = kpss(series.dropna(), regression="c", nlags="auto")
            return {
                "Series": label,
                "ADF Stat": adf_stat,
                "ADF p-value": adf_p,
                "KPSS Stat": kpss_stat,
                "KPSS p-value": kpss_p,
            }

        rows = [stationarity_row(daily, "Level"), stationarity_row(daily.diff(), "1st Difference")]
        st.dataframe(pd.DataFrame(rows).set_index("Series").style.format("{:.4f}"), use_container_width=True)
        st.caption("ADF null = unit root (non-stationary); KPSS null = stationary. Both confirm the series is stationary.")

    with col_b:
        st.subheader("3. ACF & PACF Analysis (Notebook Plot)")
        img_acf = FIGURES_DIR / "30_1_7_acf_pacf_analysis.png"
        if img_acf.exists():
            st.image(str(img_acf), use_container_width=True)
            st.caption("Figure from Notebook Cell 30: Autocorrelation (ACF) and Partial Autocorrelation (PACF) of 1st Difference.")

# ---------------------------------------------------------------------------
# Page 3: Model Diagnostics (SARIMA & XGBoost)
# ---------------------------------------------------------------------------
elif page == "Model Diagnostics (SARIMA & XGBoost)":
    st.subheader("1. SARIMA Residual Diagnostics (Notebook Plot)")
    img_sarima_diag = FIGURES_DIR / "39_1_10_sarima_model.png"
    if img_sarima_diag.exists():
        st.image(str(img_sarima_diag), use_container_width=True)
        st.caption("Figure from Notebook Cell 39: SARIMA residual diagnostic plots.")

    st.subheader("2. SARIMA Forecast vs Actuals (Notebook Plot)")
    img_sarima_fc = FIGURES_DIR / "41_1_10_sarima_model.png"
    if img_sarima_fc.exists():
        st.image(str(img_sarima_fc), use_container_width=True)
        st.caption("Figure from Notebook Cell 41: SARIMA validation and test set predictions.")

    st.divider()

    st.subheader("3. XGBoost Feature Importances (Notebook Plot)")
    img_xgb_feat = FIGURES_DIR / "49_1_11_machine_learning_model_xgboost.png"
    if img_xgb_feat.exists():
        st.image(str(img_xgb_feat), use_container_width=True)
        st.caption("Figure from Notebook Cell 49: XGBoost feature importances and lag contribution.")

    st.subheader("4. XGBoost Validation & Test Forecasts (Notebook Plots)")
    col1, col2 = st.columns(2)
    with col1:
        img_xgb_val = FIGURES_DIR / "51_1_11_machine_learning_model_xgboost.png"
        if img_xgb_val.exists():
            st.image(str(img_xgb_val), use_container_width=True)
            st.caption("Figure from Notebook Cell 51: XGBoost validation performance.")
    with col2:
        img_xgb_test = FIGURES_DIR / "52_1_11_machine_learning_model_xgboost.png"
        if img_xgb_test.exists():
            st.image(str(img_xgb_test), use_container_width=True)
            st.caption("Figure from Notebook Cell 52: XGBoost test set evaluation.")

# ---------------------------------------------------------------------------
# Page 4: Model Comparison
# ---------------------------------------------------------------------------
elif page == "Model Comparison":
    st.subheader("1. Model Performance Summary Plot (Notebook Plot)")
    img_comp = FIGURES_DIR / "57_1_12_model_comparison_evaluation.png"
    if img_comp.exists():
        st.image(str(img_comp), use_container_width=True)
        st.caption("Figure from Notebook Cell 57: Comparative evaluation metrics (RMSE, MAE, MAPE) and fitted curves across models.")

    st.subheader("2. Detailed Performance Metrics Table")
    model_order = ["Naive", "Seasonal Naive (7d)", "SARIMA", "XGBoost"]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Validation Set Metrics**")
        val_df = pd.DataFrame(metrics["validation"]).T.loc[model_order].sort_values("RMSE")
        st.dataframe(val_df.style.format("{:.2f}"), use_container_width=True)
    with col_b:
        st.markdown("**Test Set Metrics**")
        test_df = pd.DataFrame(metrics["test"]).T.loc[model_order].sort_values("RMSE")
        st.dataframe(test_df.style.format("{:.2f}"), use_container_width=True)

# ---------------------------------------------------------------------------
# Page 5: Forecast Explorer
# ---------------------------------------------------------------------------
elif page == "Forecast Explorer":
    st.subheader("Out-of-Sample Horizon Forecast Explorer")
    st.caption("Interactive projection beyond observed history built from trained SARIMA & XGBoost models.")

    horizon = st.slider("Days to forecast ahead", min_value=7, max_value=90, value=30, step=1)
    future_index = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")

    with st.spinner("Generating forecast projections..."):
        from statsmodels.tsa.statespace.sarimax import SARIMAXResults
        from xgboost import XGBRegressor

        sarima_deploy = SARIMAXResults.load(str(MODELS_DIR / "sarima_deploy.pickle"))
        exog_future = fourier_terms(pd.DatetimeIndex(list(daily.index) + list(future_index))).loc[future_index]
        sarima_future = sarima_deploy.get_forecast(steps=horizon, exog=exog_future)
        sarima_future_mean = pd.Series(sarima_future.predicted_mean.values, index=future_index)
        ci = sarima_future.conf_int(alpha=0.2)
        sarima_lower = pd.Series(ci.iloc[:, 0].values, index=future_index)
        sarima_upper = pd.Series(ci.iloc[:, 1].values, index=future_index)

        with open(MODELS_DIR / "xgb_best_iteration.json") as f:
            best_iter = json.load(f)["best_iteration"]
        xgb_deploy = XGBRegressor(
            n_estimators=best_iter, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        xgb_deploy.load_model(str(MODELS_DIR / "xgb_deploy.json"))
        xgb_future = recursive_forecast(xgb_deploy, daily, future_index)

    fig = go.Figure()
    recent = daily.iloc[-90:]
    fig.add_trace(go.Scatter(x=recent.index, y=recent.values, mode="lines", name="Recent history", line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=future_index, y=sarima_upper.values, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=future_index,
            y=sarima_lower.values,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255, 127, 14, 0.2)",
            line=dict(width=0),
            name="SARIMA 80% interval",
            hoverinfo="skip",
        )
    )
    fig.add_trace(go.Scatter(x=future_index, y=sarima_future_mean.values, mode="lines", name="SARIMA forecast", line=dict(color="#ff7f0e", width=2.2)))
    fig.add_trace(go.Scatter(x=future_index, y=xgb_future.values, mode="lines", name="XGBoost forecast", line=dict(color="#2ca02c", width=2.2)))
    fig.update_layout(height=460, margin=dict(l=15, r=15, t=15, b=15), yaxis_title="Ozone (µg/m³)")
    st.plotly_chart(fig, use_container_width=True)

    forecast_table = pd.DataFrame(
        {
            "Date": future_index.date,
            "SARIMA forecast": sarima_future_mean.values,
            "SARIMA lower (80%)": sarima_lower.values,
            "SARIMA upper (80%)": sarima_upper.values,
            "XGBoost forecast": xgb_future.values,
        }
    )
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)
    st.download_button("Download forecast as CSV", forecast_table.to_csv(index=False), file_name="ozone_forecast.csv", mime="text/csv")
