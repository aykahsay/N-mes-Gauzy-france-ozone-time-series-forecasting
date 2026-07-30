# 🌍 Paris Ground-Level Ozone ($O_3$) Time Series Forecasting

> **Time Series Analysis and Forecasting of Ground-Level Ozone ($O_3$) Concentrations in Paris Using Statistical and Machine Learning Models with an Interactive Streamlit Dashboard.**

---

## 📌 Project Overview

Ground-level ozone ($O_3$) is a major atmospheric pollutant affecting public health and environmental quality. This project delivers an end-to-end time series analysis and forecasting framework for daily mean ozone levels in Paris. 

Key workflow components:
- **Data Acquisition & Preprocessing**: Ingested hourly OpenAQ sensor measurements for Paris, handled missing observations, and aggregated readings into a continuous daily mean time series.
- **Exploratory Data Analysis & Stationarity Testing**: Evaluated seasonal patterns, autocorrelation (ACF/PACF), STL decomposition, and unit root tests (ADF and KPSS).
- **Forecasting Models**: Benchmark comparison between baseline models (Naive, 7-day Seasonal Naive), Statistical SARIMA models, and Gradient-Boosted Decision Trees (XGBoost) with engineered calendar, Fourier, and lag features.
- **Interactive Web App**: A multi-tab Streamlit dashboard providing dynamic visualizations, metric comparisons, and out-of-sample forecast simulations under custom scenarios.

---

## 📁 Repository Structure

```text
paris-ozone-time-series-forecasting/
│
├── dashboard/
│   └── app.py                     # Streamlit dashboard application
│
├── data/
│   ├── processed/                 # Cleaned and processed daily datasets
│   │   ├── daily.csv
│   │   └── predictions.csv
│   └── raw/
│
├── models/                        # Serialized models and evaluation artifacts
│   ├── metrics.json               # Validation and Test performance metrics
│   ├── sarima_deploy.pickle       # Trained SARIMA model checkpoint
│   ├── sarima_order.json          # SARIMA model orders: (0,0,0)x(1,0,1,7)
│   ├── xgb_best_iteration.json    # Optimal boosting iterations for XGBoost
│   └── xgb_deploy.json            # Serialized XGBoost model weights
│
├── src/                           # Modular Python source package
│   ├── __init__.py
│   ├── data.py                    # Data loading & preprocessing routines
│   ├── features.py                # Feature engineering (lags, rolling stats, Fourier terms)
│   └── train.py                   # Model training and artifact export pipeline
│
├── notebooks/                     # Exploratory notebooks and analyses
│   └── paris-ozone-time-series-forecasting.ipynb
│
├── ozone_df.csv                   # Raw extracted OpenAQ dataset
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git ignore file
└── README.md                      # Project documentation
```

---

## 📊 Model Benchmark & Results

Models were evaluated across validation and out-of-sample test splits using Mean Absolute Error (**MAE**), Root Mean Squared Error (**RMSE**), and Mean Absolute Percentage Error (**MAPE**).

### Test Set Performance Summary

| Model | MAE ($\mu g/m^3$) | RMSE ($\mu g/m^3$) | MAPE (%) |
| :--- | :---: | :---: | :---: |
| 🥇 **SARIMA $(0,0,0) \times (1,0,1)_7$** | **14.38** | **17.57** | **29.47%** |
| 🥈 **XGBoost (with Lags & Fourier)** | 19.35 | 23.79 | 31.74% |
| 🥉 **Naive Baseline** | 29.32 | 33.62 | 46.62% |
| 4️⃣ **Seasonal Naive (7-Day)** | 38.93 | 43.33 | 62.63% |

*SARIMA demonstrated superior generalization on unseen test data, capturing weekly seasonality and autoregressive decay patterns effectively.*

---

## 🖥️ Streamlit Interactive Dashboard

The project includes an interactive web interface built with **Streamlit** and **Plotly**.

### Key Features:
1. **📈 Data & Stationarity Analysis**: Interactive time series plot, STL trend/seasonal decomposition, ACF/PACF plots, and automated ADF/KPSS test results.
2. **⚖️ Model Performance Benchmark**: Side-by-side metric tables (MAE, RMSE, MAPE) and dynamic forecast overlays for Train, Validation, and Test splits.
3. **🔮 Out-of-Sample Forecasting**: Interactive 1 to 90-day future forecasting tool with user-defined temperature/humidity delta sliders for scenario simulation.

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup

Ensure you have **Python 3.9+** installed. Clone the repository and install required packages:

```bash
# Clone repository
git clone https://github.com/aykahsay/paris-ozone-time-series-forecasting.git
cd paris-ozone-time-series-forecasting

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Retrain Pipeline & Export Artifacts (Optional)

To rerun the automated data ingestion, feature engineering, and model fitting pipeline:

```bash
python -m src.train
```

### 3. Launch Interactive Dashboard

To start the Streamlit web application locally:

```bash
streamlit run dashboard/app.py
```

The application will be accessible in your web browser at `http://localhost:8501`.

---

## 🛠️ Built With

- **Python 3.10+**
- **Statsmodels**: SARIMA, ADF, KPSS, STL Decomposition
- **XGBoost**: Gradient Boosted Decision Trees
- **Pandas & NumPy**: Data processing and matrix operations
- **Plotly**: Interactive visualization charts
- **Streamlit**: Web dashboard framework

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
