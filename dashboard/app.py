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

st.set_page_config(page_title="Paris Ozone Forecasting", page_icon="🌫️", layout="wide")

MODEL_COLORS = {
    "Actual": "#1a1a1a",
    "Naive": "#9e9e9e",
    "Seasonal Naive (7d)": "#c2a83e",
    "SARIMA": "#e0672b",
    "XGBoost": "#2f8f5e",
}

BASE_FONT_SIZE = 15
CHART_FONT = dict(family="Inter, Segoe UI, sans-serif", size=13, color="#31333F")

# ---------------------------------------------------------------------------
# Visibility: larger base text, bigger metric tiles, higher-contrast captions,
# a slightly wider sidebar. Uses Streamlit's own CSS variables where possible
# so it still adapts if the viewer is on a dark theme.
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    html, body, [class*="st-emotion-cache"] {{ font-size: {BASE_FONT_SIZE}px; }}
    section[data-testid="stSidebar"] {{ min-width: 300px; }}
    [data-testid="stMetricValue"] {{ font-size: 1.9rem; font-weight: 700; }}
    [data-testid="stMetricLabel"] {{ font-size: 0.95rem; opacity: 0.85; }}
    [data-testid="stCaptionContainer"] p, .stCaption {{
        font-size: 0.95rem !important;
        opacity: 0.95 !important;
    }}
    h1 {{ font-size: 2.1rem !important; }}
    h2, .stSubheader {{ font-size: 1.4rem !important; margin-top: 0.6rem; }}
    section[data-testid="stSidebar"] .stRadio label p {{ font-size: 1.05rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_fig(fig, height=420, showlegend=True):
    """Apply a consistent, higher-contrast look to every Plotly chart."""
    fig.update_layout(
        height=height,
        showlegend=showlegend,
        margin=dict(l=10, r=10, t=40 if fig.layout.title.text else 10, b=10),
        font=CHART_FONT,
        legend=dict(font=dict(size=13), orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(font_size=13),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)", tickfont=dict(size=12))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)", tickfont=dict(size=12),
                      title_font=dict(size=13))
    return fig


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

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌫️ Ozone Forecasting")
    st.caption("Paris, ground-level O₃")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Overview", "Decomposition & Stationarity", "Model Comparison", "Forecast Explorer"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        f"**Data range**  \n{daily.index.min().date()} → {daily.index.max().date()}  \n"
        f"({len(daily)} calendar days)"
    )
    st.caption(
        f"**Long sensor gaps**  \n{gap_mask.sum()} days interpolated ({gap_mask.mean():.0%})"
    )

st.title("Paris Ozone (O₃) Forecasting Dashboard")
st.caption(
    f"Daily mean ground-level ozone, {daily.index.min().date()} to {daily.index.max().date()} "
    f"({len(daily)} days). Built on OpenAQ hourly measurements resampled to daily means."
)

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
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
                              name="Daily mean ozone", line=dict(color=MODEL_COLORS["Actual"], width=1.4)))
    in_gap = view_gaps.values
    if in_gap.any():
        # index 0 always starts a new run (no wrap-around to the window's last element),
        # so this stays correct as the user drags the date-range slider to any sub-window.
        prev = np.concatenate(([~in_gap[0]], in_gap[:-1]))
        gap_id = (in_gap != prev).cumsum()
        for _, idx in pd.Series(range(len(view))).groupby(gap_id):
            if in_gap[idx.iloc[0]]:
                fig.add_vrect(x0=view.index[idx.iloc[0]], x1=view.index[idx.iloc[-1]],
                              fillcolor="red", opacity=0.15, line_width=0)
    style_fig(fig, showlegend=False)
    fig.update_layout(yaxis_title="Ozone (µg/m³)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Shaded red bands mark days that fall inside a long (>14 day) sensor outage and were "
               "linearly interpolated — treat values there as synthetic, not measured.")

    st.subheader("Summary Statistics")
    st.dataframe(daily.describe().to_frame("Ozone (µg/m³)").T, width="stretch")

# ---------------------------------------------------------------------------
# Page: Decomposition & Stationarity
# ---------------------------------------------------------------------------
elif page == "Decomposition & Stationarity":
    st.subheader("STL Decomposition (annual period = 365 days)")
    stl_result = STL(daily, period=365, robust=True).fit()

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                         subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
    fig.add_trace(go.Scatter(x=daily.index, y=daily.values, mode="lines",
                              line=dict(width=1.2, color=MODEL_COLORS["Actual"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily.index, y=stl_result.trend, mode="lines",
                              line=dict(width=2, color=MODEL_COLORS["SARIMA"])), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily.index, y=stl_result.seasonal, mode="lines",
                              line=dict(width=1, color=MODEL_COLORS["XGBoost"])), row=3, col=1)
    fig.add_trace(go.Scatter(x=daily.index, y=stl_result.resid, mode="markers",
                              marker=dict(size=3, color="#666")), row=4, col=1)
    style_fig(fig, height=680, showlegend=False)
    st.plotly_chart(fig, width="stretch")
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
        st.dataframe(pd.DataFrame(rows).set_index("Series").style.format("{:.4f}"), width="stretch")
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
        fig2.add_trace(go.Bar(x=list(range(nlags + 1)), y=acf_vals, marker_color=MODEL_COLORS["XGBoost"]), row=1, col=1)
        fig2.add_trace(go.Bar(x=list(range(nlags + 1)), y=pacf_vals, marker_color=MODEL_COLORS["SARIMA"]), row=1, col=2)
        for col in (1, 2):
            fig2.add_hline(y=conf, line_dash="dot", line_color="gray", row=1, col=col)
            fig2.add_hline(y=-conf, line_dash="dot", line_color="gray", row=1, col=col)
        style_fig(fig2, height=380, showlegend=False)
        st.plotly_chart(fig2, width="stretch")

# ---------------------------------------------------------------------------
# Page: Model Comparison
# ---------------------------------------------------------------------------
elif page == "Model Comparison":
    st.subheader("Validation vs. Test Metrics")
    model_order = ["Naive", "Seasonal Naive (7d)", "SARIMA", "XGBoost"]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Validation set**")
        val_df = pd.DataFrame(metrics["validation"]).T.loc[model_order].sort_values("RMSE")
        st.dataframe(val_df.style.format("{:.2f}"), width="stretch")
    with col_b:
        st.markdown("**Test set**")
        test_df = pd.DataFrame(metrics["test"]).T.loc[model_order].sort_values("RMSE")
        st.dataframe(test_df.style.format("{:.2f}"), width="stretch")

    metric_choice = st.radio("Metric", ["RMSE", "MAE", "MAPE (%)"], horizontal=True)
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(name="Validation", x=model_order, y=val_df.loc[model_order, metric_choice],
                              marker_color="#8ec3b0"))
    bar_fig.add_trace(go.Bar(name="Test", x=model_order, y=test_df.loc[model_order, metric_choice],
                              marker_color=MODEL_COLORS["SARIMA"]))
    style_fig(bar_fig, height=380)
    bar_fig.update_layout(barmode="group", yaxis_title=metric_choice)
    st.plotly_chart(bar_fig, width="stretch")

    st.subheader("Forecasts vs. Actual (Validation + Test Period)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions["Actual"], mode="lines",
                              name="Actual", line=dict(color=MODEL_COLORS["Actual"], width=1.8)))
    for model_name in ["SARIMA", "XGBoost"]:
        fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions[model_name], mode="lines",
                                  name=model_name, line=dict(color=MODEL_COLORS[model_name], width=1.6)))
    split_boundary = predictions.loc[predictions["split"] == "test", "Date"].min()
    fig.add_vline(x=split_boundary, line_dash="dash", line_color="gray")
    style_fig(fig)
    fig.update_layout(yaxis_title="Ozone (µg/m³)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Dashed line marks the validation/test boundary. MAPE is unstable on validation because "
               "several winter days have near-zero actual ozone; MAE/RMSE are the more reliable metrics.")

# ---------------------------------------------------------------------------
# Page: Forecast Explorer
# ---------------------------------------------------------------------------
elif page == "Forecast Explorer":
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
                              name="Recent history", line=dict(color=MODEL_COLORS["Actual"], width=1.6)))
    fig.add_trace(go.Scatter(x=future_index, y=sarima_upper.values, mode="lines",
                              line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_index, y=sarima_lower.values, mode="lines", fill="tonexty",
                              fillcolor="rgba(224,103,43,0.18)", line=dict(width=0),
                              name="SARIMA 80% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_index, y=sarima_future_mean.values, mode="lines",
                              name="SARIMA forecast", line=dict(color=MODEL_COLORS["SARIMA"], width=2)))
    fig.add_trace(go.Scatter(x=future_index, y=xgb_future.values, mode="lines",
                              name="XGBoost forecast", line=dict(color=MODEL_COLORS["XGBoost"], width=2)))
    style_fig(fig, height=460)
    fig.update_layout(yaxis_title="Ozone (µg/m³)")
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "These are genuine out-of-sample projections beyond the last observed date "
        f"({daily.index.max().date()}) — there is no ground truth to compare against yet. "
        "Both models were only ever validated on the historical val/test split shown in the "
        "'Model Comparison' page; treat accuracy here as declining with horizon length, especially "
        "past ~30 days, since neither model has weather covariates to anchor the annual cycle precisely."
    )

    forecast_table = pd.DataFrame({
        "Date": future_index.date,
        "SARIMA forecast": sarima_future_mean.values,
        "SARIMA lower (80%)": sarima_lower.values,
        "SARIMA upper (80%)": sarima_upper.values,
        "XGBoost forecast": xgb_future.values,
    })
    st.dataframe(forecast_table, width="stretch", hide_index=True)
    st.download_button("Download forecast as CSV", forecast_table.to_csv(index=False),
                        file_name="ozone_forecast.csv", mime="text/csv")
