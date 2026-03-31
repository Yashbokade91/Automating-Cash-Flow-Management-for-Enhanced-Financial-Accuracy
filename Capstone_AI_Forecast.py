"""
Capstone Appendix Script: AI Forecast for Cash Flow & KPI Comparison
Author: Yash Bokade
Project: Automating Cash Flow Management for Enhanced Financial Accuracy

What this script does
---------------------
1) Loads the project Excel ("Data_base.xlsx").
2) Trains a simple, transparent model (Linear Regression) on BEFORE data to predict Net Cash
   using Inflows and Outflows. (You can swap to ARIMA or XGBoost later—keep it reproducible.)
3) Generates an AI_Forecast for the AFTER period.
4) Computes accuracy metrics and summary KPIs.
5) Saves an output workbook with predictions & KPIs, and exports three Matplotlib charts as PNGs.

Why Linear Regression?
----------------------
- Fast, explainable, easy to defend in an academic setting.
- Coefficients show how Inflows/Outflows relate to Net Cash.
- Avoids "black box" concerns for a management-focused capstone.

How to run
----------
$ pip install pandas scikit-learn matplotlib openpyxl
$ python capstone_ai_forecast_appendix.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# --- Paths ---
BASE = Path(".")
excel_in = BASE / "Data_base.xlsx"
excel_out = BASE / "Results.xlsx"

# --- Helper functions ---
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (0..2). Lower is better."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1e-9
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom)

def accuracy_from_error(y_true, y_pred):
    """Turn an error metric into an accuracy-like number (1 - normalized error)."""
    # We'll use normalized MAE over IQR to be robust to outliers.
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # FIX: Explicitly cast to float to prevent tuple unpacking ValueErrors in older NumPy versions
    q75 = float(np.percentile(y_true, 75))
    q25 = float(np.percentile(y_true, 25))
    
    iqr = max(q75 - q25, 1e-9)
    norm_mae = mae / iqr
    acc = max(0.0, 1.0 - norm_mae)  # clamp to [0, 1]
    return acc

def load_sheets(path):
    before = pd.read_excel(path, sheet_name="Data_base_Before")
    after  = pd.read_excel(path, sheet_name="Data_base_After")
    summary = pd.read_excel(path, sheet_name="Summary")
    # Ensure Date is datetime
    for df in (before, after):
        if not np.issubdtype(before["Date"].dtype, np.datetime64):
            df["Date"] = pd.to_datetime(df["Date"])
    return before, after, summary


def train_model(before_df):
    # Ensure data is sorted chronologically
    df = before_df.sort_values("Date").copy()

    # Create lagged features (t-1) to act as independent variables
    df["Inflows_Lag1"] = df["Inflows ($)"].shift(1)
    df["Outflows_Lag1"] = df["Outflows ($)"].shift(1)

    # Drop the first row, which will have NaNs due to the shift
    df = df.dropna(subset=["Inflows_Lag1", "Outflows_Lag1", "Net Cash ($)"])

    # Define features (X) as historical data and target (y) as current Net Cash
    X = df[["Inflows_Lag1", "Outflows_Lag1"]]
    y = df["Net Cash ($)"]

    model = LinearRegression()
    model.fit(X, y)
    return model


def generate_predictions(model, df):
    # Ensure data is sorted chronologically
    df_sorted = df.sort_values("Date").copy()

    # Create lagged features (t-1)
    df_sorted["Inflows_Lag1"] = df_sorted["Inflows ($)"].shift(1)
    df_sorted["Outflows_Lag1"] = df_sorted["Outflows ($)"].shift(1)

    # FIX: Use dropna() instead of bfill() to prevent data leakage from the current row
    df_valid = df_sorted.dropna(subset=["Inflows_Lag1", "Outflows_Lag1"]).copy()

    X = df_valid[["Inflows_Lag1", "Outflows_Lag1"]]
    preds = model.predict(X)

    # Return a pandas Series aligned with the valid index
    return pd.Series(preds, index=df_valid.index)

def main():
    before, after, summary = load_sheets(excel_in)

    # Train on BEFORE period
    model = train_model(before)

    # Save coefficients for interpretability
    intercept = float(model.intercept_)
    coef_inflows, coef_outflows = [float(c) for c in model.coef_]

    # AI forecast on AFTER period (you can also do full-period prediction if you want)
    after_pred = generate_predictions(model, after)
    after["AI Forecast (Model)"] = after_pred

    # FIX: Drop NaNs from the metric calculation to account for the first dropped row
    valid_after = after.dropna(subset=["AI Forecast (Model)", "Net Cash ($)"])

    # Compute accuracy metrics using the valid rows
    ai_acc = accuracy_from_error(valid_after["Net Cash ($)"], valid_after["AI Forecast (Model)"])
    manual_acc = None
    if "Manual Forecast ($)" in valid_after.columns:
        manual_acc = accuracy_from_error(valid_after["Net Cash ($)"], valid_after["Manual Forecast ($)"])

    # Summary KPIs (Before vs After from the workbook)
    kpi_rows = []
    kpi_rows.append({
        "Metric": "AI Forecast Accuracy (model-derived)",
        "Before Automation": np.nan,
        "After Automation": round(ai_acc, 4),
        "Improvement (%)": np.nan
    })
    if manual_acc is not None:
        kpi_rows.append({
            "Metric": "Manual Forecast Accuracy (recomputed)",
            "Before Automation": round(manual_acc, 4),
            "After Automation": round(manual_acc, 4),  # same metric definition
            "Improvement (%)": np.nan
        })
    kpi_df = pd.DataFrame(kpi_rows)

    # Export combined results
    with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
        before.to_excel(writer, sheet_name="Data_base_Before", index=False)
        after.to_excel(writer, sheet_name="Data_base_After_with_AI", index=False)
        summary.to_excel(writer, sheet_name="Summary_Original", index=False)
        kpi_df.to_excel(writer, sheet_name="AI_Model_KPIs", index=False)

    # --- Charts ---
    # 1) Forecast Accuracy: Manual vs AI (bar)
    labels = []
    values = []
    if manual_acc is not None:
        labels.append("Manual")
        values.append(manual_acc)
    labels.append("AI (Model)")
    values.append(ai_acc)

    plt.figure()
    plt.bar(labels, values)
    plt.title("Forecast Accuracy: Manual vs AI (Model)")
    plt.ylabel("Accuracy (0..1)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("Fig1_ForecastAccuracy.png", dpi=200)
    plt.close()

    # 2) Line chart: Net Cash vs Manual vs AI for AFTER period
    plot_df = after.copy()
    plot_df = plot_df.sort_values("Date")
    plt.figure()
    plt.plot(plot_df["Date"], plot_df["Net Cash ($)"], label="Net Cash (Actual)")
    if "Manual Forecast ($)" in plot_df.columns:
        plt.plot(plot_df["Date"], plot_df["Manual Forecast ($)"], label="Manual Forecast")
    plt.plot(plot_df["Date"], plot_df["AI Forecast (Model)"], label="AI Forecast (Model)")
    plt.title("Actual vs Forecast (After Period)")
    plt.xlabel("Date")
    plt.ylabel("Amount ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Fig2_Actual_vs_Forecast_After.png", dpi=200)
    plt.close()

    # 3) Before vs After Posting Time (bar) — from original Summary means if available
    try:
        row = summary[summary["Metric"].str.contains("Posting Time", case=False)].iloc[0]
        before_time = float(row["Before Automation"])
        after_time = float(row["After Automation"])
        plt.figure()
        plt.bar(["Before", "After"], [before_time, after_time])
        plt.title("Financial Posting/Reconciliation Time: Before vs After")
        plt.ylabel("Avg Minutes")
        plt.tight_layout()
        plt.savefig("Fig3_Time_Before_After.png", dpi=200)
        plt.close()
    except Exception as e:
        # If summary row isn't available, compute from raw
        before_time = before["Posting Time (min)"].mean()
        after_time = after["Posting Time (min)"].mean()
        plt.figure()
        plt.bar(["Before", "After"], [before_time, after_time])
        plt.title("Financial Posting/Reconciliation Time: Before vs After")
        plt.ylabel("Avg Minutes")
        plt.tight_layout()
        plt.savefig("Fig3_Time_Before_After.png", dpi=200)
        plt.close()

    # Print simple model card (optional console output)
    print("Linear Regression Model")
    print(f"Net Cash = {intercept:.2f} + {coef_inflows:.4f}*Inflows + {coef_outflows:.4f}*Outflows")
    print(f"Manual Accuracy (recomputed): {manual_acc}")
    print(f"AI Accuracy (model): {ai_acc}")

if __name__ == "__main__":
    main()
