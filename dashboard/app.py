"""
Streamlit interactive dashboard for the Paris Ozone Forecasting Project.

Includes dedicated pages for Dataset KPIs & Summary Statistics, Time Series Overview,
Decomposition & Stationarity, Model Diagnostics, Model Comparison, and Forecast Explorer.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import fourier_terms, recursive_forecast  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "dashboard" / "figures"

st.set_page_config(page_title="Paris Ozone Forecasting", page_icon="🌫️", layout="wide")

# Custom styling for high readability
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { min-width: 320px; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #1f77b4; }
    [data-testid="stMetricLabel"] { font-size: 0.9rem; font-weight: 600; opacity: 0.9; }
    .kpi-card {
        background-color: rgba(31, 119, 180, 0.08);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        border-left: 4px solid #1f77b4;
    }
    .stCaption { font-size: 0.92rem !important; opacity: 0.9 !important; }
    h1 { font-size: 2.1rem !important; }
    h2, .stSubheader { font-size: 1.35rem !important; margin-top: 0.6rem; }
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

# ---------------------------------------------------------------------------
# Sidebar: Navigation & Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌫️ Paris Ozone Dashboard")
    st.caption("STA4030 Time Series Term Paper")
    st.divider()

    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Page Selection",
        [
            "KPIs & Summary Statistics",
            "Overview & Time Series",
            "Decomposition & Stationarity",
            "Model Diagnostics & Training",
            "Model Comparison",
            "Forecast Explorer",
        ],
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown("### ⚙️ Date Filter Window")
    date_filter = st.slider(
        "Filter Date Range",
        min_value=daily.index.min().to_pydatetime(),
        max_value=daily.index.max().to_pydatetime(),
        value=(daily.index.min().to_pydatetime(), daily.index.max().to_pydatetime()),
        format="YYYY-MM",
    )
    st.caption(f"📅 Selected: {date_filter[0].strftime('%Y-%m-%d')} → {date_filter[1].strftime('%Y-%m-%d')}")

# Filter dataset according to sidebar slider
filtered_daily = daily.loc[date_filter[0] : date_filter[1]]
filtered_gaps = gap_mask.loc[date_filter[0] : date_filter[1]]

# Main Header
st.title("Paris Ozone (O₃) Time Series Forecasting Dashboard")
st.caption(
    f"Interactive analysis of daily mean ground-level ozone in Paris ({daily.index.min().date()} to {daily.index.max().date()})."
)

# ---------------------------------------------------------------------------
# Dedicated Page 1: KPIs & Summary Statistics (Dataset Focus)
# ---------------------------------------------------------------------------
if page == "KPIs & Summary Statistics":
    st.subheader("📊 Dataset Key Performance Indicators (KPIs)")

    # Row 1: Summary Statistics KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Mean Ozone", f"{filtered_daily.mean():.2f} µg/m³")
    c2.metric("Median Ozone", f"{filtered_daily.median():.2f} µg/m³")
    c3.metric("Max Concentration", f"{filtered_daily.max():.2f} µg/m³")
    c4.metric("Min Concentration", f"{filtered_daily.min():.2f} µg/m³")
    c5.metric("Std Deviation", f"{filtered_daily.std():.2f} µg/m³")
    iqr_val = filtered_daily.quantile(0.75) - filtered_daily.quantile(0.25)
    c6.metric("IQR", f"{iqr_val:.2f} µg/m³")

    st.markdown("---")

    # Row 2: Data Quality & Observations KPIs
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Total Observation Days", f"{len(filtered_daily)} days")
    q2.metric("Interpolated Outage Days", f"{filtered_gaps.sum()} days")
    q3.metric("Data Gap Percentage", f"{filtered_gaps.mean():.1%}")
    q4.metric("Variance", f"{filtered_daily.var():.2f}")

    # Distribution Visualizations
    st.markdown("### 📉 Distribution & Seasonal Box Plot")
    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        st.markdown("#### Ozone Concentration Histogram & Density")
        fig_hist = px.histogram(
            filtered_daily,
            x=filtered_daily.values,
            nbins=40,
            title="Distribution of Daily Mean Ozone",
            labels={"x": "Ozone (µg/m³)"},
            color_discrete_sequence=["#1f77b4"],
            marginal="box",
        )
        fig_hist.update_layout(height=380, margin=dict(l=15, r=15, t=35, b=15))
        st.plotly_chart(fig_hist, use_container_width=True)

    with fig_col2:
        st.markdown("#### Monthly Seasonal Variation (Box Plot)")
        df_box = filtered_daily.to_frame("Ozone")
        df_box["Month"] = df_box.index.strftime("%b")
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        df_box["Month"] = pd.Categorical(df_box["Month"], categories=month_order, ordered=True)
        df_box = df_box.sort_values("Month")

        fig_box = px.box(df_box, x="Month", y="Ozone", title="Monthly Ozone Distributions", color_discrete_sequence=["#ff7f0e"])
        fig_box.update_layout(height=380, margin=dict(l=15, r=15, t=35, b=15), yaxis_title="Ozone (µg/m³)")
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    st.markdown("### 📈 Comprehensive Statistical Summary")
    col_t1, col_t2 = st.columns([1, 1])

    with col_t1:
        st.markdown("#### Descriptors & Statistical Moments")
        stats_series = filtered_daily.describe()
        stats_series["variance"] = filtered_daily.var()
        stats_series["skewness"] = filtered_daily.skew()
        stats_series["kurtosis"] = filtered_daily.kurtosis()

        stats_df = stats_series.to_frame("Ozone Value (µg/m³)")
        st.dataframe(stats_df.style.format("{:.3f}"), use_container_width=True)

        st.download_button(
            "📥 Download Summary Stats CSV",
            stats_df.to_csv(),
            file_name="paris_ozone_summary_statistics.csv",
            mime="text/csv",
        )

    with col_t2:
        st.markdown("#### Monthly Aggregated Statistics")
        monthly_df = filtered_daily.to_frame("Ozone")
        monthly_df["Month"] = monthly_df.index.strftime("%Y-%m (%b)")
        monthly_summary = (
            monthly_df.groupby("Month")["Ozone"]
            .agg(["mean", "std", "min", "max", "count"])
            .rename(columns={"mean": "Mean", "std": "Std Dev", "min": "Min", "max": "Max", "count": "Days"})
        )
        st.dataframe(monthly_summary.style.format({"Mean": "{:.2f}", "Std Dev": "{:.2f}", "Min": "{:.2f}", "Max": "{:.2f}"}), use_container_width=True, height=340)

# ---------------------------------------------------------------------------
# Page 2: Overview & Time Series
# ---------------------------------------------------------------------------
elif page == "Overview & Time Series":
    st.subheader("Selected Window Summary KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Filtered Window Mean", f"{filtered_daily.mean():.1f} µg/m³")
    k2.metric("Filtered Window Max", f"{filtered_daily.max():.1f} µg/m³")
    k3.metric("Filtered Days Count", f"{len(filtered_daily)} days")
    k4.metric("Outage Days in Window", f"{filtered_gaps.sum()} days ({filtered_gaps.mean():.1%})")

    view_mode = st.radio("View Mode", ["Interactive Plotly Chart", "Original Notebook Plots"], horizontal=True)

    if view_mode == "Interactive Plotly Chart":
        st.markdown("#### Dynamic Interactive Time Series")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=filtered_daily.index,
                y=filtered_daily.values,
                mode="lines",
                name="Daily Mean Ozone",
                line=dict(color="#1f77b4", width=1.8),
                hovertemplate="%{x|%Y-%m-%d}: <b>%{y:.2f} µg/m³</b><extra></extra>",
            )
        )

        # Highlight gap outages
        in_gap = filtered_gaps.values
        if in_gap.any():
            prev = np.concatenate(([~in_gap[0]], in_gap[:-1]))
            gap_id = (in_gap != prev).cumsum()
            for _, idx in pd.Series(range(len(filtered_daily))).groupby(gap_id):
                if in_gap[idx.iloc[0]]:
                    fig.add_vrect(
                        x0=filtered_daily.index[idx.iloc[0]],
                        x1=filtered_daily.index[idx.iloc[-1]],
                        fillcolor="rgba(239, 85, 59, 0.25)",
                        line_width=0,
                    )

        fig.update_layout(
            height=450,
            xaxis_title="Date",
            yaxis_title="Ozone Concentration (µg/m³)",
            margin=dict(l=15, r=15, t=20, b=15),
            hovermode="x unified",
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red shaded vertical regions indicate long sensor outage periods (>14 days) interpolated linearly.")

    else:
        st.markdown("#### Original Notebook Exploratory Plots")
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            img_eda = FIGURES_DIR / "22_1_4_handling_missing_data_resampling.png"
            if img_eda.exists():
                st.image(str(img_eda), use_container_width=True)
                st.caption("Figure from Notebook Cell 22: Daily mean ozone series showing interpolated gaps.")
        with col_n2:
            img_raw = FIGURES_DIR / "17_1_3_exploratory_data_analysis.png"
            if img_raw.exists():
                st.image(str(img_raw), use_container_width=True)
                st.caption("Figure from Notebook Cell 17: Raw hourly measurements and distribution.")

# ---------------------------------------------------------------------------
# Page 3: Decomposition & Stationarity
# ---------------------------------------------------------------------------
elif page == "Decomposition & Stationarity":
    st.subheader("Seasonal-Trend Decomposition & Stationarity Tests")

    tab_inter, tab_nb = st.tabs(["⚡ Interactive Analysis", "📓 Original Notebook Figures"])

    with tab_inter:
        from statsmodels.tsa.seasonal import STL

        stl_res = STL(daily, period=365, robust=True).fit()

        fig_stl = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=("Observed Data", "Trend Component", "Seasonal Component (365d)", "Residual Noise")
        )
        fig_stl.add_trace(go.Scatter(x=daily.index, y=daily.values, mode="lines", line=dict(color="#1f77b4", width=1.4)), row=1, col=1)
        fig_stl.add_trace(go.Scatter(x=daily.index, y=stl_res.trend, mode="lines", line=dict(color="#ff7f0e", width=2)), row=2, col=1)
        fig_stl.add_trace(go.Scatter(x=daily.index, y=stl_res.seasonal, mode="lines", line=dict(color="#2ca02c", width=1.4)), row=3, col=1)
        fig_stl.add_trace(go.Scatter(x=daily.index, y=stl_res.resid, mode="markers", marker=dict(size=2.5, color="#7f7f7f")), row=4, col=1)
        fig_stl.update_layout(height=650, showlegend=False, margin=dict(l=15, r=15, t=30, b=15))
        st.plotly_chart(fig_stl, use_container_width=True)

        col_st1, col_st2 = st.columns(2)
        with col_st1:
            st.markdown("#### Stationarity Hypothesis Tests")
            from statsmodels.tsa.stattools import adfuller, kpss

            def get_stat_row(series, label):
                adf_stat, adf_p, *_ = adfuller(series.dropna(), autolag="AIC")
                kpss_stat, kpss_p, *_ = kpss(series.dropna(), regression="c", nlags="auto")
                return {
                    "Series": label,
                    "ADF Stat": adf_stat,
                    "ADF p-val": adf_p,
                    "KPSS Stat": kpss_stat,
                    "KPSS p-val": kpss_p,
                }

            stat_df = pd.DataFrame([get_stat_row(daily, "Level (Raw)"), get_stat_row(daily.diff(), "1st Difference")])
            st.dataframe(stat_df.set_index("Series").style.format("{:.4f}"), use_container_width=True)
            st.caption("ADF test null = Non-stationary. KPSS test null = Stationary. Both confirm level series is stationary.")

        with col_st2:
            st.markdown("#### Interactive ACF / PACF Lags")
            from statsmodels.graphics.tsaplots import acf as acf_values, pacf as pacf_values

            nlags = st.slider("Select ACF/PACF Lags", min_value=10, max_value=60, value=30)
            diffed = daily.diff().dropna()
            acf_vals = acf_values(diffed, nlags=nlags)
            pacf_vals = pacf_values(diffed, nlags=nlags)
            conf = 1.96 / np.sqrt(len(diffed))

            fig_ap = make_subplots(rows=1, cols=2, subplot_titles=("Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"))
            fig_ap.add_trace(go.Bar(x=list(range(nlags + 1)), y=acf_vals, marker_color="#2ca02c"), row=1, col=1)
            fig_ap.add_trace(go.Bar(x=list(range(nlags + 1)), y=pacf_vals, marker_color="#ff7f0e"), row=1, col=2)
            for c in (1, 2):
                fig_ap.add_hline(y=conf, line_dash="dot", line_color="gray", row=1, col=c)
                fig_ap.add_hline(y=-conf, line_dash="dot", line_color="gray", row=1, col=c)
            fig_ap.update_layout(height=350, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_ap, use_container_width=True)

    with tab_nb:
        st.markdown("#### Original Notebook Figures")
        img_stl = FIGURES_DIR / "24_1_5_seasonal_trend_decomposition_stl.png"
        if img_stl.exists():
            st.image(str(img_stl), use_container_width=True)
            st.caption("Figure from Notebook Cell 24: STL Decomposition with 365-day annual period.")

        img_acf = FIGURES_DIR / "30_1_7_acf_pacf_analysis.png"
        if img_acf.exists():
            st.image(str(img_acf), use_container_width=True)
            st.caption("Figure from Notebook Cell 30: Autocorrelation (ACF) & Partial Autocorrelation (PACF).")

# ---------------------------------------------------------------------------
# Page 4: Model Diagnostics & Training
# ---------------------------------------------------------------------------
elif page == "Model Diagnostics & Training":
    st.subheader("Model Diagnostic & Feature Importance Plots")

    st.markdown("#### Notebook Residual & Feature Figures")
    c_d1, c_d2 = st.columns(2)
    with c_d1:
        img_sarima_diag = FIGURES_DIR / "39_1_10_sarima_model.png"
        if img_sarima_diag.exists():
            st.image(str(img_sarima_diag), use_container_width=True)
            st.caption("Figure from Notebook Cell 39: SARIMA Residual Diagnostics.")

        img_sarima_fc = FIGURES_DIR / "41_1_10_sarima_model.png"
        if img_sarima_fc.exists():
            st.image(str(img_sarima_fc), use_container_width=True)
            st.caption("Figure from Notebook Cell 41: SARIMA Predictions vs Actuals.")

    with c_d2:
        img_xgb_feat = FIGURES_DIR / "49_1_11_machine_learning_model_xgboost.png"
        if img_xgb_feat.exists():
            st.image(str(img_xgb_feat), use_container_width=True)
            st.caption("Figure from Notebook Cell 49: XGBoost Feature Importance Ranking.")

        img_xgb_val = FIGURES_DIR / "51_1_11_machine_learning_model_xgboost.png"
        if img_xgb_val.exists():
            st.image(str(img_xgb_val), use_container_width=True)
            st.caption("Figure from Notebook Cell 51: XGBoost Validation Performance.")

# ---------------------------------------------------------------------------
# Page 5: Model Comparison (Model Performance KPIs Featured Here)
# ---------------------------------------------------------------------------
elif page == "Model Comparison":
    st.subheader("🏆 Model Performance KPIs & Comparative Evaluation")

    best_val_model = min(metrics["validation"], key=lambda k: metrics["validation"][k]["RMSE"])
    best_test_model = min(metrics["test"], key=lambda k: metrics["test"][k]["RMSE"])

    # Model Performance KPIs Banner
    kpi_m1, kpi_m2, kpi_m3, kpi_m4 = st.columns(4)
    kpi_m1.metric("Best Val Model", best_val_model, f"RMSE: {metrics['validation'][best_val_model]['RMSE']:.2f}")
    kpi_m2.metric("Best Test Model", best_test_model, f"RMSE: {metrics['test'][best_test_model]['RMSE']:.2f}")
    kpi_m3.metric("Best Test MAE", f"{metrics['test'][best_test_model]['MAE']:.2f} µg/m³")
    kpi_m4.metric("Best Test MAPE", f"{metrics['test'][best_test_model]['MAPE (%)']:.2f}%")

    st.markdown("---")

    tab_c1, tab_c2 = st.tabs(["📊 Interactive Metrics Comparison", "📓 Notebook Evaluation Summary Plot"])

    with tab_c1:
        model_order = ["Naive", "Seasonal Naive (7d)", "SARIMA", "XGBoost"]

        col_v, col_t = st.columns(2)
        with col_v:
            st.markdown("#### Validation Set Performance Table")
            val_df = pd.DataFrame(metrics["validation"]).T.loc[model_order].sort_values("RMSE")
            st.dataframe(val_df.style.format("{:.2f}"), use_container_width=True)

        with col_t:
            st.markdown("#### Test Set Performance Table")
            test_df = pd.DataFrame(metrics["test"]).T.loc[model_order].sort_values("RMSE")
            st.dataframe(test_df.style.format("{:.2f}"), use_container_width=True)

        metric_select = st.radio("Compare Metric Across Models", ["RMSE", "MAE", "MAPE (%)"], horizontal=True)

        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(name="Validation", x=model_order, y=val_df.loc[model_order, metric_select], marker_color="#1f77b4"))
        bar_fig.add_trace(go.Bar(name="Test", x=model_order, y=test_df.loc[model_order, metric_select], marker_color="#ff7f0e"))
        bar_fig.update_layout(barmode="group", height=380, yaxis_title=metric_select, margin=dict(l=15, r=15, t=30, b=15))
        st.plotly_chart(bar_fig, use_container_width=True)

        st.markdown("#### Interactive Predictions vs Ground Truth")
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=predictions["Date"], y=predictions["Actual"], mode="lines", name="Actual", line=dict(color="#1f77b4", width=2)))
        for m in ["SARIMA", "XGBoost"]:
            fig_pred.add_trace(go.Scatter(x=predictions["Date"], y=predictions[m], mode="lines", name=m))
        split_bound = predictions.loc[predictions["split"] == "test", "Date"].min()
        fig_pred.add_vline(x=split_bound, line_dash="dash", line_color="gray", annotation_text="Validation / Test Split")
        fig_pred.update_layout(height=420, yaxis_title="Ozone (µg/m³)", margin=dict(l=15, r=15, t=20, b=15))
        st.plotly_chart(fig_pred, use_container_width=True)

    with tab_c2:
        img_comp = FIGURES_DIR / "57_1_12_model_comparison_evaluation.png"
        if img_comp.exists():
            st.image(str(img_comp), use_container_width=True)
            st.caption("Figure from Notebook Cell 57: Comprehensive performance comparison plot from the paper.")

# ---------------------------------------------------------------------------
# Page 6: Forecast Explorer (Best Performing Model Only)
# ---------------------------------------------------------------------------
elif page == "Forecast Explorer":
    st.subheader("Interactive Out-of-Sample Horizon Forecast Explorer")

    # Determine best model dynamically based on lowest Test RMSE
    best_test_model = min(metrics["test"], key=lambda k: metrics["test"][k]["RMSE"])
    best_rmse = metrics["test"][best_test_model]["RMSE"]
    best_mae = metrics["test"][best_test_model]["MAE"]
    best_mape = metrics["test"][best_test_model]["MAPE (%)"]

    st.success(
        f"🏆 **Champion Model Selected**: **{best_test_model}** (Evaluated as the best model with lowest Test RMSE = **{best_rmse:.2f}**, MAE = **{best_mae:.2f}** µg/m³, MAPE = **{best_mape:.2f}%**)"
    )

    col_h1, col_h2 = st.columns([2, 1])
    with col_h1:
        horizon = st.slider("Select Forecast Horizon (Days Ahead)", min_value=7, max_value=90, value=30, step=1)
    with col_h2:
        confidence = st.selectbox(f"{best_test_model} Prediction Interval", [80, 90, 95], index=0)

    future_index = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    alpha_val = 1 - (confidence / 100.0)

    with st.spinner(f"Refitting champion model ({best_test_model}) and projecting future horizon..."):
        from statsmodels.tsa.statespace.sarimax import SARIMAXResults

        sarima_deploy = SARIMAXResults.load(str(MODELS_DIR / "sarima_deploy.pickle"))
        exog_future = fourier_terms(pd.DatetimeIndex(list(daily.index) + list(future_index))).loc[future_index]
        sarima_future = sarima_deploy.get_forecast(steps=horizon, exog=exog_future)
        sarima_future_mean = pd.Series(sarima_future.predicted_mean.values, index=future_index)
        ci = sarima_future.conf_int(alpha=alpha_val)
        sarima_lower = pd.Series(ci.iloc[:, 0].values, index=future_index)
        sarima_upper = pd.Series(ci.iloc[:, 1].values, index=future_index)

    fig_fc = go.Figure()
    recent = daily.iloc[-90:]
    fig_fc.add_trace(go.Scatter(x=recent.index, y=recent.values, mode="lines", name="Recent Observed History", line=dict(color="#1f77b4", width=2)))
    fig_fc.add_trace(go.Scatter(x=future_index, y=sarima_upper.values, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_fc.add_trace(
        go.Scatter(
            x=future_index,
            y=sarima_lower.values,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255, 127, 14, 0.22)",
            line=dict(width=0),
            name=f"{best_test_model} {confidence}% Interval",
            hoverinfo="skip",
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=future_index,
            y=sarima_future_mean.values,
            mode="lines",
            name=f"{best_test_model} Forecast (Champion)",
            line=dict(color="#ff7f0e", width=2.5),
        )
    )
    fig_fc.update_layout(height=460, margin=dict(l=15, r=15, t=20, b=15), yaxis_title="Ozone Concentration (µg/m³)")
    st.plotly_chart(fig_fc, use_container_width=True)

    st.caption(
        f"Out-of-sample projections up to {horizon} days beyond the observed period ({daily.index.max().date()}). "
        f"Generated using the top-performing {best_test_model} model."
    )

    forecast_table = pd.DataFrame(
        {
            "Date": future_index.date,
            f"{best_test_model} Forecast": sarima_future_mean.values,
            f"Lower ({confidence}%)": sarima_lower.values,
            f"Upper ({confidence}%)": sarima_upper.values,
        }
    )

    st.markdown("#### Detailed Forecast Data Table")
    st.dataframe(
        forecast_table.style.format(
            {f"{best_test_model} Forecast": "{:.2f}", f"Lower ({confidence}%)": "{:.2f}", f"Upper ({confidence}%)": "{:.2f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "📥 Download Forecast CSV",
        forecast_table.to_csv(index=False),
        file_name=f"paris_ozone_{best_test_model.lower()}_forecast_{horizon}d.csv",
        mime="text/csv",
    )
