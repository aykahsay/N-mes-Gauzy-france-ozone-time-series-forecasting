# 🌍 Nîmes Ground-Level Ozone ($O_3$) Time Series Forecasting & Analysis

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/XGBoost-2.0%2B-green.svg)](https://xgboost.readthedocs.io/)
[![Time Series](https://img.shields.io/badge/Statsmodels-0.14%2B-orange.svg)](https://www.statsmodels.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aykahsay/paris-ozone-time-series-forecasting/blob/main/nimes_ozone_time_series_forecasting.ipynb)

> **An End-to-End Time Series Analysis, Statistical Modeling, Machine Learning Forecasting, and Interactive Dashboard Framework for Daily Ground-Level Ozone ($O_3$) Concentrations in Nîmes (Les Hauts de Nîmes), Occitanie, France.**
> 
> *STA4030 Time Series Analysis & Forecasting — Term Paper & Project Repository*  
> **Authors:** Faith Chakwanira, Ruth Musahnu, Ambachow Kahsay, Justice Chawanda, Francis Kinyanjui, Cynthia Gathogo  
> **Lecturer:** Prof. Alex Wambua

---

## 📖 Table of Contents

- [📌 Project Overview](#-project-overview)
- [📄 Term Paper Summary](#-term-paper-summary)
- [📓 Introduction to the Jupyter Notebook (`nimes_ozone_time_series_forecasting.ipynb`)](#-introduction-to-the-jupyter-notebook-nimes_ozone_time_series_forecastingipynb)
  - [Notebook Architecture & Section-by-Section Walkthrough](#notebook-architecture--section-by-section-walkthrough)
  - [Key Theoretical & Diagnostic Intuitions](#key-theoretical--diagnostic-intuitions)
- [📊 Dataset & Data Auditing](#-dataset--data-auditing)
- [🔬 Methodological Framework & Stationarity Diagnosis](#-methodological-framework--stationarity-diagnosis)
- [🏆 Model Benchmark & Performance Evaluation](#-model-benchmark--performance-evaluation)
- [🖥️ Streamlit Interactive Web Application](#️-streamlit-interactive-web-application)
- [📁 Repository Structure](#-repository-structure)
- [🚀 Quick Start & Execution Guide](#-quick-start--execution-guide)
- [📜 License & Citation](#-license--citation)

---

## 📌 Project Overview

Ground-level ozone ($O_3$) is a hazardous secondary atmospheric pollutant formed photochemically when nitrogen oxides ($NO_x$) and volatile organic compounds ($VOCs$) react under intense solar radiation and elevated temperatures. Unlike primary pollutants emitted directly from tailpipes or smokestacks, ambient ozone spikes are tightly coupled with summer heatwaves and atmospheric stagnations.

This repository provides a comprehensive time-series analysis and forecasting system for daily mean ground-level ozone at the **Nîmes Gauzy** station (Sensor `7774985`, OpenAQ API v3) in Occitanie, southern France:

1. **Rigorous Data Quality Control & Anomaly Auditing**: Identification and masking of a 21-day all-zero instrument fault block in July 2025.
2. **Exploratory Data Analysis (EDA) & Regulatory Profiling**: Analysis of diurnal cycles, monthly climatologies, and reconstruction of Maximum Daily 8-Hour Averages (MDA8) against European Union air quality standards.
3. **Formal Stationarity & Unit Root Hypothesis Testing**: Combined Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) testing to differentiate between stochastic and deterministic seasonality.
4. **Statistical & Machine Learning Forecasting Benchmark**: Evaluation of Baseline models (Naive, 7-Day Seasonal Naive), Statistical SARIMA with Fourier harmonic terms, and Gradient-Boosted Decision Trees (XGBoost).
5. **Interactive Streamlit Web Dashboard**: Dynamic multi-tab web application for real-time visualization, metric exploration, and scenario-based multi-step out-of-sample forecasting.

---

## 📄 Term Paper Summary

The accompanying PDF report, [STA4030 Term Paper _group_work.pdf](file:///c:/School_projects/STA4030_time_series/termpaper/paris-ozone-time-series-forecasting/STA4030%20Term%20Paper%20_group_work.pdf), presents the complete academic write-up submitted for STA4030 Time Series Analysis & Forecasting under the supervision of Prof. Alex Wambua.

### Key Takeaways from the Paper:
* **Instrument Fault Impact**: Failing to mask the 21-day July 2025 all-zero instrument outage artificially deflates the July monthly mean from **$81.8\ \mu g/m^3$ to $62.9\ \mu g/m^3$**, creating a false climatological dip during peak ozone season.
* **Deterministic Seasonality Verdict**: ADF rejects a unit root ($t = -3.871, p < 0.01$) while KPSS rejects level stationarity ($LM = 0.918, p < 0.01$). This joint result proves the series possesses a deterministic annual mean shift, favoring Fourier harmonic regression over seasonal differencing.
* **Why SARIMA Beats XGBoost on Long Horizons**: Harmonic SARIMA achieves a test MAE of **$20.89\ \mu g/m^3$** (RMSE $24.63$), outperforming XGBoost (test MAE **$35.76\ \mu g/m^3$**). Feature gain analysis reveals $89.4\%$ of XGBoost's weight concentrates on short-term lags (1-day lag = $0.699$), which decay into flat line predictions during multi-step recursive forecasting.
* **Regulatory Exceedance Floor**: Reconstructing the MDA8 metric revealed **33 days exceeding the EU $120\ \mu g/m^3$ target value** across the record—with 21 exceedance days occurring in the first seven months of 2026 alone, already breaching the annual statutory limit of 18 days.

---

## 📓 Introduction to the Jupyter Notebook (`nimes_ozone_time_series_forecasting.ipynb`)

The primary analysis file, [nimes_ozone_time_series_forecasting.ipynb](file:///c:/School_projects/STA4030_time_series/termpaper/paris-ozone-time-series-forecasting/nimes_ozone_time_series_forecasting.ipynb), is a self-contained, fully reproducible Google Colab / Jupyter notebook structured into **16 analytical sections**. 

### Notebook Architecture & Section-by-Section Walkthrough

```text
nimes_ozone_time_series_forecasting.ipynb
├── 1. Data Collection (OpenAQ v3 REST API)
├── 2. Load Data & Metadata Verification
│   ├── 2.1 Dataset Metadata Inspection
│   └── 2.2 Data Structures & Consistency Checks
├── 3. Exploratory Data Analysis (EDA)
├── 4. Handling Missing Data & Resampling
│   ├── 4.1 Missing-Value Map & Instrument Outage Audit
│   └── 4.2 Supplementary Visualizations (Scatter, Climatology, Candlestick)
├── 5. Seasonal-Trend Decomposition (STL)
├── 6. Stationarity Testing (ADF & KPSS)
├── 7. Autocorrelation (ACF & PACF) Analysis
├── 8. Chronological Train / Validation / Test Split
├── 9. Baseline Forecasting Models
├── 10. Statistical Model: SARIMA with Harmonic Fourier Terms
├── 11. Machine Learning Model: XGBoost Regression
├── 12. Side-by-Side Model Comparison & Evaluation
├── 13. Discussion & Technical Findings
├── 14. Interactive Dashboard Integration
├── 15. Recommendations & Future Enhancements
└── 16. Conclusion
```

#### Detailed Section Breakdown & Intuition:

1. **Section 1–2: Data Ingestion & Metadata Auditing**
   * *Intuition*: Ingests raw hourly OpenAQ measurements (`sensor 7774985`, Nîmes Gauzy) spanning 28 January 2024 to 1 August 2026 (14,398 raw observations).
   * *Key Check*: Promotes UTC timestamps to datetime index, verifies absence of duplicate timestamps or negative concentration values, and inspects zero distributions.

2. **Section 3–4: Quality Screening & Missing Value Imputation**
   * *Intuition*: Identifies 359 hourly readings of `0.0 µg/m³` falling inside a single 21-day block (3 to 26 July 2025). Midday zero ozone in southern France is physically impossible due to solar photochemistry.
   * *Treatment*: Masks the 21-day fault block as missing, performs time-based linear interpolation, and flags synthetic days in a boolean mask. Resamples hourly data to daily means ($917$ calendar days).

3. **Section 5–7: Time Series Decomposition & Stationarity Diagnosis**
   * *Intuition*: Evaluates underlying trend and seasonal signals. STL decomposition isolates a dominant annual cycle (period = 365 days).
   * *Stationarity Intuition*: Evaluates level vs. differenced series using ADF and KPSS tests:
     $$\text{ADF Level: } t = -3.871 \ (< -3.44 \text{ critical value}) \implies \text{Reject Unit Root}$$
     $$\text{KPSS Level: } LM = 0.918 \ (> 0.739 \text{ critical value}) \implies \text{Reject Stationarity}$$
     *Takeaway*: The level reverts to a mean within seasons, but the mean itself shifts deterministically by month. Differencing is unnecessary and introduces artificial negative lag-1 autocorrelation ($r = -0.26$).

4. **Section 8: Chronological Train / Validation / Test Split**
   * *Intuition*: Avoids random data splitting (which causes data leakage in time series). Uses chronological splitting:
     * **Training Set**: 551 days (28 Jan 2024 to 31 Jul 2025, 60%)
     * **Validation Set**: 183 days (1 Aug 2025 to 30 Jan 2026, 20%) — used exclusively for hyperparameter tuning & early stopping.
     * **Test Set**: 183 days (31 Jan 2026 to 1 Aug 2026, 20%) — held-out, evaluated exactly once.

5. **Section 9–11: Model Fitting & Hyperparameter Tuning**
   * **Naive Baselines**: Last-observed value ($y_t = y_{t-1}$) and 7-day Seasonal Naive ($y_t = y_{t-7}$).
   * **SARIMA with Fourier Terms**: Grid searches $(p,d,q) \times (P,D,Q)_7$ orders using exogenous sine/cosine pairs ($\sin(2\pi k t / 365), \cos(2\pi k t / 365)$). Selects optimal order **SARIMA $(2,1,0) \times (1,0,1)_7$**.
   * **XGBoost Regression**: Features include 6 lag terms (1, 2, 3, 7, 14, 21 days), rolling 7-day and 14-day statistics, calendar features, and Fourier terms. Uses early stopping on validation RMSE (optimal boosting rounds = $71$).

6. **Section 12–16: Evaluation, Discussion & Dashboard Integration**
   * *Intuition*: Compares predictions on validation and test sets using MAE, RMSE, and MAPE. Explains why harmonic statistical models provide superior generalization over recursive decision trees for multi-step univariate horizons.

---

## 📊 Dataset & Data Auditing

| Field / Attribute | Value | Details / Notes |
| :--- | :--- | :--- |
| **Location** | Nîmes Gauzy (ID `2162655`) | Nîmes, Occitanie, France ($43.85^{\circ}\text{N}, 4.37^{\circ}\text{E}$) |
| **Sensor / Parameter** | Sensor `7774985` · $O_3$ mass | OpenAQ v3 REST API |
| **Units** | $\mu g/m^3$ | Mass concentration |
| **Raw Resolution** | Hourly, irregularly spaced | 14,398 raw observations (~80% 1h step, remainder $\ge 2\text{h}$) |
| **Modeling Grid** | Daily mean | 917 continuous calendar days (28 Jan 2024 to 1 Aug 2026) |
| **Usable Observations** | 14,039 hourly readings | 359 zero readings on 21 instrument fault days masked & interpolated |
| **Concentration Range** | $0.0$ to $221.4\ \mu g/m^3$ | Mean: $61.59\ \mu g/m^3$, Median: $62.3\ \mu g/m^3$, SD: $31.20\ \mu g/m^3$ |
| **EU Threshold Breaches** | 5 hours $> 180\ \mu g/m^3$ | Information threshold breached; 0 alert breaches ($>240\ \mu g/m^3$) |
| **MDA8 Exceedance Days** | 33 days $> 120\ \mu g/m^3$ | 2 days in 2024, 10 in 2025, **21 in first 7 months of 2026** |

---

## 🔬 Methodological Framework & Stationarity Diagnosis

### Why Deterministic Seasonality Matters
Standard Box-Jenkins methodology defaults to seasonal differencing ($\Delta_7$ or $\Delta_{365}$) whenever seasonality is present. However, differencing a series with **deterministic seasonality** introduces an invertible moving-average unit root ($MA(1)$ with $\theta \approx 1$), leading to over-differencing.

In our analysis:
1. **Level ADF Test**: $t = -3.871 \implies p < 0.01$ (Unit root rejected; level mean-reverts).
2. **Level KPSS Test**: $LM = 0.918 \implies p < 0.01$ (Stationarity rejected; level varies deterministically by month).
3. **Differenced Series ACF**: $r_1 = -0.26$, confirming over-differencing.

**Solution**: Model seasonality deterministically using **annual Fourier terms**:
$$x_{k,t}^{(s)} = \sin\left(\frac{2\pi k t}{365.25}\right), \quad x_{k,t}^{(c)} = \cos\left(\frac{2\pi k t}{365.25}\right) \quad \text{for } k \in \{1, 2\}$$

---

## 🏆 Model Benchmark & Performance Evaluation

Models were evaluated on held-out validation (183 days) and test (183 days) splits using Mean Absolute Error (**MAE**), Root Mean Squared Error (**RMSE**), and Mean Absolute Percentage Error (**MAPE**).

### Final Benchmark Performance Summary

| Model | Validation MAE ($\mu g/m^3$) | Validation RMSE ($\mu g/m^3$) | Test MAE ($\mu g/m^3$) | Test RMSE ($\mu g/m^3$) | Test MAPE (%) | Performance Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **SARIMA $(2,1,0) \times (1,0,1)_7$ (with Fourier)** | **13.82** | **16.91** | **20.89** | **24.63** | **26.36%** | **Best Overall** |
| 🥈 **XGBoost (with Lags & Fourier)** | 14.90 | 18.46 | 35.76 | 40.58 | 44.14% | Competitive Val / Degraded Test |
| 🥉 **Seasonal Naive (7-Day Persistence)** | 30.76 | 36.22 | 37.35 | 43.21 | 46.81% | Baseline Floor |
| 4️⃣ **Naive Baseline (Last Value)** | 30.09 | 34.87 | 51.97 | 55.73 | 65.80% | Unusable Long-Horizon |

*Note: MAPE is inflated by near-zero winter observations in the denominator; MAE and RMSE are the authoritative evaluation metrics.*

---

## 🖥️ Streamlit Interactive Web Application

The project includes a multi-page interactive web application located in `dashboard/app.py`.

### Key Features:
1. **📊 KPIs & Summary Statistics**: Full descriptive moments, missing data gap rates, monthly aggregation tables, interactive distribution histograms, and monthly seasonal boxplots.
2. **📈 Time Series Overview & Stationarity**: Interactive Plotly time series chart, STL trend/seasonal decomposition, ACF/PACF plots, and automated ADF/KPSS test results.
3. **🏆 Model Performance Benchmark**: Side-by-side metric tables (MAE, RMSE, MAPE), performance KPI cards, and dynamic forecast overlay charts.
4. **🔮 Champion Model Out-of-Sample Forecasting**: Interactive 7 to 90-day future forecasting engine powered by the trained **SARIMA** model with dynamic prediction intervals ($80\%$ and $95\%$) and CSV exports.

---

## 📁 Repository Structure

```text
paris-ozone-time-series-forecasting/
│
├── dashboard/
│   ├── app.py                             # Streamlit web application
│   └── figures/                           # Exported diagnostic charts
│
├── data/
│   └── processed/                         # Automated pipeline outputs
│       ├── daily.csv                      # Cleaned daily series & gap mask
│       └── predictions.csv                # Actual vs. model predictions
│
├── models/                                # Serialized model checkpoints
│   ├── metrics.json                       # Validation & Test error metrics
│   ├── sarima_deploy.pickle               # Trained deployment SARIMA model
│   ├── sarima_order.json                  # Selected SARIMA orders: (2,1,0)x(1,0,1,7)
│   ├── xgb_best_iteration.json            # Optimal boosting rounds (71)
│   └── xgb_deploy.json                    # Serialized XGBoost model weights
│
├── src/                                   # Modular Python package
│   ├── __init__.py
│   ├── data.py                            # Data loading & preprocessing module
│   ├── features.py                        # Feature engineering (lags, Fourier, rolling stats)
│   └── train.py                           # Automated model training pipeline
│
├── LesHautsdeNîmes_ozone_df.csv           # Primary OpenAQ raw dataset for Nîmes
├── STA4030 Term Paper _group_work.pdf     # Final group term paper PDF report
├── nimes_ozone_time_series_forecasting.ipynb # Complete analysis & forecasting notebook
├── script.Rmd                             # R Markdown analysis script
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore rules
└── README.md                              # Project documentation
```

---

## 🚀 Quick Start & Execution Guide

### 1. Prerequisites & Installation

Ensure **Python 3.9+** is installed on your system. Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/aykahsay/paris-ozone-time-series-forecasting.git
cd paris-ozone-time-series-forecasting

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook

Open and execute the analysis notebook locally or in VS Code / Jupyter Lab:

```bash
jupyter notebook nimes_ozone_time_series_forecasting.ipynb
```

Or launch directly in **Google Colab**:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aykahsay/paris-ozone-time-series-forecasting/blob/main/nimes_ozone_time_series_forecasting.ipynb)

### 3. Retrain the Automated Pipeline

To rerun data cleaning, hyperparameter optimization, and model serialization:

```bash
python -m src.train
```

### 4. Launch the Interactive Web Dashboard

To launch the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

Access the dashboard in your web browser at `http://localhost:8501`.

---

## 📜 License & Citation

Distributed under the **MIT License**. See `LICENSE` for details.

### Citation
If you use this repository or term paper in your research or coursework, please cite:

```bibtex
@techreport{chakwanira2026nimes,
  title     = {Forecasting Ground-Level Ozone in N\^imes: A univariate study of 14,039 hourly O3 observations from a southern French monitoring station, and the limits of what they can support},
  author    = {Chakwanira, Faith and Musahnu, Ruth and Kahsay, Ambachow and Chawanda, Justice and Kinyanjui, Francis and Gathogo, Cynthia},
  institution = {School of Mathematics, STA4030 Time Series Analysis},
  supervisor= {Prof. Alex Wambua},
  year      = {2026},
  month     = {August}
}
```
