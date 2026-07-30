"""
Streamlit dashboard for the Paris ozone forecasting project.

Reads precomputed artifacts from data/processed/ and models/ (produced by
`python -m src.train`) so the app starts instantly instead of refitting the
SARIMA grid search on every reload. Run with:

    streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import acf as acf_values, pacf as pacf_values
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import fourier_terms, recursive_forecast, FEATURE_COLS  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="Paris Ozone Forecasting", layout="wide")

MODEL_COLORS = {
    "Actual": "#111111",
    "Naive": "#9e9e9e",
    "Seasonal Naive (7d)": "#c2a83e",
    "SARIMA": "#e07b39",
    "XGBoost": "#3b8f6b",
}


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------
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


@st.cache_resource
def load_sarima_deploy():
    return SARIMAXResults.load(str(MODELS_DIR / "sarima_deploy.pickle"))


@st.cache_resource
def load_xgb_deploy():
    with open(MODELS_DIR / "xgb_best_iteration.json") as f:
        best_iter = json.load(f)["best_iteration"]
    model = XGBRegressor(n_estimators=best_iter, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.load_model(str(MODELS_DIR / "xgb_deploy.json"))
    return model


ARTIFACTS_MISSING = not (PROCESSED_DIR / "daily.csv").exists()

if ARTIFACTS_MISSING:
    st.error(
        "No trained artifacts found. Run `python -m src.train` from the project root first "
        "to fit the models and generate data/processed/ and models/ before launching this dashboard."
    )
    st.stop()

daily, gap_mask = load_daily()
predictions = load_predictions()
metrics = load_metrics()

st.title("Paris Ozone (O₃) Forecasting Dashboard")
st.caption(
    f"Daily mean ground-level ozone, {daily.index.min().date()} to {daily.index.max().date()} "
    f"({len(daily)} days). Built on OpenAQ hourly measurements resampled to daily means."
)

tab_overview, tab_decomp, tab_models, tab_forecast = st.tabs(
    ["Overview", "Decomposition & Stationarity", "Model Comparison", "Forecast Explorer"]
)

# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Ozone (µg/m³)", f"{daily.mean():.1f}")
    col2.metric("Max Ozone (µg/m³)", f"{daily.max():.1f}")
    col3.metric("Calendar Days", f"{len(daily)}")
    col4.metric("Interpolated (long-gap) Days", f"{gap_mask.sum()} ({gap_mask.mean():.0%})")

    st.subheader("Daily Mean Ozone Concentration")
    date_range = st.slider(
        "Date range",
        min_value=daily.index.min().to_pydatetime(),
        max_value=daily.index.max().to_pydatetime(),
        value=(daily.index.min().to_pydatetime(), daily.index.max().to_pydatetime()),
    )
    view = daily.loc[date_range[0]:date_range[1]]
    view_gaps = gap_mask.loc[date_range[0]:date_range[1]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.index, y=view.values, mode="lines",
                              name="Daily mean ozone", line=dict(color=MODEL_COLORS["Actual"], width=1)))
    in_gap = view_gaps.values
    if in_gap.any():
        # index 0 always starts a new run (no wrap-around to the window's last element),
        # so this stays correct as the user drags the date-range slider to any sub-window.
        prev = np.concatenate(([~in_gap[0]], in_gap[:-1]))
        gap_id = (in_gap != prev).cumsum()
        for _, idx in pd.Series(range(len(view))).groupby(gap_id):
            if in_gap[idx.iloc[0]]:
                fig.add_vrect(x0=view.index[idx.iloc[0]], x1=view.index[idx.iloc[-1]],
                              fillcolor="red", opacity=0.12, line_width=0)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Ozone (µg/m³)",
                       showlegend=False)
    st.plotly_chart(fig, width='stretch')
    st.caption("Shaded red bands mark days that fall inside a long (>14 day) sensor outage and were "
               "linearly interpolated — treat values there as synthetic, not measured.")

    st.subheader("Summary Statistics")
    st.dataframe(daily.describe().to_frame("Ozone (µg/m³)").T, width='stretch')

# ---------------------------------------------------------------------------
# Tab 2: Decomposition & Stationarity
# ---------------------------------------------------------------------------
with tab_decomp:
    st.subheader("STL Decomposition (annual period = 365 days)")
    stl_result = STL(daily, period=365, robust=True).fit()

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                         subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
    fig.add_trace(go.Scatter(x=daily.index, y=daily.values, mode="lines", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily.index, y=stl_result.trend, mode="lines", line=dict(width=1.5, color="#e07b39")), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily.index, y=stl_result.seasonal, mode="lines", line=dict(width=1, color="#3b8f6b")), row=3, col=1)
    fig.add_trace(go.Scatter(x=daily.index, y=stl_result.resid, mode="markers", marker=dict(size=3, color="#888")), row=4, col=1)
    fig.update_layout(height=650, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width='stretch')
    st.caption("Trend shows a mild increase over the observation window; seasonal confirms the expected "
               "summer-peak / winter-trough annual cycle; residual captures weather-driven day-to-day noise.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Stationarity Tests")
        def stationarity_row(series, label):
            adf_stat, adf_p, *_ = adfuller(series.dropna(), autolag="AIC")
            kpss_stat, kpss_p, *_ = kpss(series.dropna(), regression="c", nlags="auto")
            return {"Series": label, "ADF stat": adf_stat, "ADF p-value": adf_p,
                    "KPSS stat": kpss_stat, "KPSS p-value": kpss_p}
        rows = [stationarity_row(daily, "Level"), stationarity_row(daily.diff(), "1st difference")]
        st.dataframe(pd.DataFrame(rows).set_index("Series"), width='stretch')
        st.caption("ADF null = unit root (non-stationary); KPSS null = stationary. Both agree the level "
                   "series is already stationary.")

    with col_b:
        st.subheader("ACF / PACF (1st difference)")
        diffed = daily.diff().dropna()
        nlags = 40
        acf_vals = acf_values(diffed, nlags=nlags)
        pacf_vals = pacf_values(diffed, nlags=nlags)
        conf = 1.96 / np.sqrt(len(diffed))
        fig2 = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
        fig2.add_trace(go.Bar(x=list(range(nlags + 1)), y=acf_vals, marker_color="#3b8f6b"), row=1, col=1)
        fig2.add_trace(go.Bar(x=list(range(nlags + 1)), y=pacf_vals, marker_color="#e07b39"), row=1, col=2)
        for col in (1, 2):
            fig2.add_hline(y=conf, line_dash="dot", line_color="gray", row=1, col=col)
            fig2.add_hline(y=-conf, line_dash="dot", line_color="gray", row=1, col=col)
        fig2.update_layout(height=350, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, width='stretch')

# ---------------------------------------------------------------------------
# Tab 3: Model Comparison
# ---------------------------------------------------------------------------
with tab_models:
    st.subheader("Validation vs. Test Metrics")
    model_order = ["Naive", "Seasonal Naive (7d)", "SARIMA", "XGBoost"]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Validation set**")
        val_df = pd.DataFrame(metrics["validation"]).T.loc[model_order].sort_values("RMSE")
        st.dataframe(val_df.style.format("{:.2f}"), width='stretch')
    with col_b:
        st.markdown("**Test set**")
        test_df = pd.DataFrame(metrics["test"]).T.loc[model_order].sort_values("RMSE")
        st.dataframe(test_df.style.format("{:.2f}"), width='stretch')

    metric_choice = st.radio("Metric", ["RMSE", "MAE", "MAPE (%)"], horizontal=True)
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(name="Validation", x=model_order, y=val_df.loc[model_order, metric_choice],
                              marker_color="#8ec3b0"))
    bar_fig.add_trace(go.Bar(name="Test", x=model_order, y=test_df.loc[model_order, metric_choice],
                              marker_color="#e07b39"))
    bar_fig.update_layout(barmode="group", height=350, margin=dict(l=10, r=10, t=30, b=10),
                           yaxis_title=metric_choice)
    st.plotly_chart(bar_fig, width='stretch')

    st.subheader("Forecasts vs. Actual (Validation + Test Period)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions["Actual"], mode="lines",
                              name="Actual", line=dict(color=MODEL_COLORS["Actual"], width=1.5)))
    for model_name in ["SARIMA", "XGBoost"]:
        fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions[model_name], mode="lines",
                                  name=model_name, line=dict(color=MODEL_COLORS[model_name])))
    split_boundary = predictions.loc[predictions["split"] == "test", "Date"].min()
    fig.add_vline(x=split_boundary, line_dash="dash", line_color="gray")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Ozone (µg/m³)")
    st.plotly_chart(fig, width='stretch')
    st.caption("Dashed line marks the validation/test boundary. MAPE is unstable on validation because "
               "several winter days have near-zero actual ozone; MAE/RMSE are the more reliable metrics.")

# ---------------------------------------------------------------------------
# Tab 4: Forecast Explorer
# ---------------------------------------------------------------------------
with tab_forecast:
    st.subheader("Forecast Beyond the Observed History")
    st.caption(
        "Both models below are refit on the full observed history (train + validation + test) "
        "using the hyperparameters selected earlier, then projected forward."
    )
    horizon = st.slider("Days to forecast ahead", min_value=7, max_value=90, value=30, step=1)

    future_index = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")

    with st.spinner("Generating forecasts..."):
        sarima_deploy = load_sarima_deploy()
        exog_future = fourier_terms(pd.DatetimeIndex(list(daily.index) + list(future_index))).loc[future_index]
        sarima_future = sarima_deploy.get_forecast(steps=horizon, exog=exog_future)
        sarima_future_mean = pd.Series(sarima_future.predicted_mean.values, index=future_index)
        ci = sarima_future.conf_int(alpha=0.2)
        sarima_lower = pd.Series(ci.iloc[:, 0].values, index=future_index)
        sarima_upper = pd.Series(ci.iloc[:, 1].values, index=future_index)

        xgb_deploy = load_xgb_deploy()
        xgb_future = recursive_forecast(xgb_deploy, daily, future_index)

    fig = go.Figure()
    recent = daily.iloc[-90:]
    fig.add_trace(go.Scatter(x=recent.index, y=recent.values, mode="lines",
                              name="Recent history", line=dict(color=MODEL_COLORS["Actual"], width=1.5)))
    fig.add_trace(go.Scatter(x=future_index, y=sarima_upper.values, mode="lines",
                              line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_index, y=sarima_lower.values, mode="lines", fill="tonexty",
                              fillcolor="rgba(224,123,57,0.15)", line=dict(width=0),
                              name="SARIMA 80% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_index, y=sarima_future_mean.values, mode="lines",
                              name="SARIMA forecast", line=dict(color=MODEL_COLORS["SARIMA"])))
    fig.add_trace(go.Scatter(x=future_index, y=xgb_future.values, mode="lines",
                              name="XGBoost forecast", line=dict(color=MODEL_COLORS["XGBoost"])))
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Ozone (µg/m³)")
    st.plotly_chart(fig, width='stretch')

    st.caption(
        "These are genuine out-of-sample projections beyond the last observed date "
        f"({daily.index.max().date()}) — there is no ground truth to compare against yet. "
        "Both models were only ever validated on the historical val/test split shown in the "
        "'Model Comparison' tab; treat accuracy here as declining with horizon length, especially "
        "past ~30 days, since neither model has weather covariates to anchor the annual cycle precisely."
    )

    forecast_table = pd.DataFrame({
        "Date": future_index.date,
        "SARIMA forecast": sarima_future_mean.values,
        "SARIMA lower (80%)": sarima_lower.values,
        "SARIMA upper (80%)": sarima_upper.values,
        "XGBoost forecast": xgb_future.values,
    })
    st.dataframe(forecast_table, width='stretch', hide_index=True)
    st.download_button("Download forecast as CSV", forecast_table.to_csv(index=False),
                        file_name="ozone_forecast.csv", mime="text/csv")
