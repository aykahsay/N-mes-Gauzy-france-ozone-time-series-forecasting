"""
Fits SARIMA and XGBoost on the train/val/test split (mirroring the analysis notebook),
evaluates both, and persists everything the dashboard needs so it never has to refit
the grid search live:

- data/processed/daily.csv         cleaned daily series + long-gap flag
- data/processed/predictions.csv   actual + val/test predictions for every model
- models/metrics.json              val/test MAE/RMSE/MAPE per model
- models/sarima_order.json         selected SARIMA order/seasonal_order
- models/sarima_deploy.pickle      SARIMA fit on the FULL history (for future forecasts)
- models/xgb_deploy.json           XGBoost fit on the FULL history (for future forecasts)
- models/xgb_best_iteration.json   selected boosting-round count
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from src.data import clean_daily, chronological_split
from src.features import fourier_terms, build_features, FEATURE_COLS, recursive_forecast

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "LesHautsdeNîmes_ozone_df.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"


def forecast_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning data...")
    daily, gap_mask = clean_daily(CSV_PATH)
    pd.DataFrame({"Ozone": daily, "long_gap": gap_mask}).to_csv(PROCESSED_DIR / "daily.csv", index_label="Date")

    train, val, test = chronological_split(daily)
    trainval = pd.concat([train, val])
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    val_results, test_results = {}, {}
    val_preds, test_preds = {}, {}

    # ---------- Baselines ----------
    def naive_forecast(history, horizon):
        return pd.Series(history.iloc[-1], index=horizon.index)

    def seasonal_naive_forecast(history, horizon, season=7):
        return pd.Series(np.resize(history.iloc[-season:].values, len(horizon)), index=horizon.index)

    val_preds["Naive"] = naive_forecast(train, val)
    test_preds["Naive"] = naive_forecast(trainval, test)
    val_preds["Seasonal Naive (7d)"] = seasonal_naive_forecast(train, val)
    test_preds["Seasonal Naive (7d)"] = seasonal_naive_forecast(trainval, test)
    for name in ["Naive", "Seasonal Naive (7d)"]:
        val_results[name] = forecast_metrics(val, val_preds[name])
        test_results[name] = forecast_metrics(test, test_preds[name])

    # ---------- SARIMA ----------
    print("Selecting SARIMA order via validation RMSE (this takes ~1 minute)...")
    exog_full = fourier_terms(daily.index)
    exog_train, exog_val, exog_test = exog_full.loc[train.index], exog_full.loc[val.index], exog_full.loc[test.index]
    exog_trainval = pd.concat([exog_train, exog_val])

    best_rmse, best_order, best_seasonal = np.inf, None, None
    for p in range(3):
        for d in [0, 1]:
            for q in range(3):
                for P in range(2):
                    for Q in range(2):
                        try:
                            fit = SARIMAX(
                                train, order=(p, d, q), seasonal_order=(P, 0, Q, 7),
                                exog=exog_train, enforce_stationarity=False, enforce_invertibility=False,
                            ).fit(disp=False, maxiter=200)
                            pred = fit.get_forecast(steps=len(val), exog=exog_val).predicted_mean
                            rmse = np.sqrt(mean_squared_error(val.values, pred.values))
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_order, best_seasonal = (p, d, q), (P, 0, Q, 7)
                        except Exception:
                            continue
    print(f"Best SARIMA order: {best_order} seasonal_order: {best_seasonal} val RMSE: {best_rmse:.2f}")
    with open(MODELS_DIR / "sarima_order.json", "w") as f:
        json.dump({"order": best_order, "seasonal_order": best_seasonal}, f)

    sarima_val_fit = SARIMAX(
        train, order=best_order, seasonal_order=best_seasonal,
        exog=exog_train, enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=200)
    sarima_val_pred = sarima_val_fit.get_forecast(steps=len(val), exog=exog_val).predicted_mean
    sarima_val_pred.index = val.index
    val_preds["SARIMA"] = sarima_val_pred
    val_results["SARIMA"] = forecast_metrics(val, sarima_val_pred)

    sarima_final_fit = SARIMAX(
        trainval, order=best_order, seasonal_order=best_seasonal,
        exog=exog_trainval, enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=200)
    sarima_test_pred = sarima_final_fit.get_forecast(steps=len(test), exog=exog_test).predicted_mean
    sarima_test_pred.index = test.index
    test_preds["SARIMA"] = sarima_test_pred
    test_results["SARIMA"] = forecast_metrics(test, sarima_test_pred)

    print("Fitting deployment SARIMA on full history...")
    exog_alldata = fourier_terms(daily.index)
    sarima_deploy_fit = SARIMAX(
        daily, order=best_order, seasonal_order=best_seasonal,
        exog=exog_alldata, enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=200)
    sarima_deploy_fit.save(str(MODELS_DIR / "sarima_deploy.pickle"))

    # ---------- XGBoost ----------
    print("Training XGBoost with early stopping on validation...")
    full_feat = build_features(daily).dropna()

    def split_xy(index):
        rows = full_feat.loc[full_feat.index.isin(index)]
        return rows[FEATURE_COLS], rows["y"]

    X_train, y_train = split_xy(train.index)
    X_val, y_val = split_xy(val.index)

    xgb_val_model = XGBRegressor(
        n_estimators=1000, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        early_stopping_rounds=30, eval_metric="rmse",
    )
    xgb_val_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best_iteration = xgb_val_model.best_iteration
    print(f"Selected boosting rounds: {best_iteration}")
    with open(MODELS_DIR / "xgb_best_iteration.json", "w") as f:
        json.dump({"best_iteration": best_iteration}, f)

    xgb_val_pred = recursive_forecast(xgb_val_model, train, val.index)
    val_preds["XGBoost"] = xgb_val_pred
    val_results["XGBoost"] = forecast_metrics(val, xgb_val_pred)

    X_trainval, y_trainval = split_xy(trainval.index)
    xgb_final_model = XGBRegressor(
        n_estimators=best_iteration, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    xgb_final_model.fit(X_trainval, y_trainval)
    xgb_test_pred = recursive_forecast(xgb_final_model, trainval, test.index)
    test_preds["XGBoost"] = xgb_test_pred
    test_results["XGBoost"] = forecast_metrics(test, xgb_test_pred)

    print("Fitting deployment XGBoost on full history...")
    X_all, y_all = split_xy(daily.index)
    xgb_deploy_model = XGBRegressor(
        n_estimators=best_iteration, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    xgb_deploy_model.fit(X_all, y_all)
    xgb_deploy_model.save_model(str(MODELS_DIR / "xgb_deploy.json"))

    # ---------- Persist predictions & metrics ----------
    pred_rows = []
    for split_name, actual, preds in [("val", val, val_preds), ("test", test, test_preds)]:
        for date in actual.index:
            row = {"Date": date, "split": split_name, "Actual": actual.loc[date]}
            for model_name, series in preds.items():
                row[model_name] = series.loc[date]
            pred_rows.append(row)
    pd.DataFrame(pred_rows).to_csv(PROCESSED_DIR / "predictions.csv", index=False)

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({"validation": val_results, "test": test_results}, f, indent=2)

    print("\nValidation results:")
    print(pd.DataFrame(val_results).T.sort_values("RMSE"))
    print("\nTest results:")
    print(pd.DataFrame(test_results).T.sort_values("RMSE"))
    print("\nDone. Artifacts written to data/processed/ and models/.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
