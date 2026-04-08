"""
generate_report.py - Generate Prediction Report from Test Data
===============================================================
Creates a comprehensive HTML report with model predictions and insights

Usage:
  python generate_report.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime

# Configuration
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("ml_artifacts")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def load_models():
    """Load all trained models and artifacts"""
    print("Loading models...")

    clf = joblib.load(ARTIFACTS_DIR / "model_shortage_classifier.pkl")
    reg_prob = joblib.load(ARTIFACTS_DIR / "model_shortage_probability.pkl")

    # Load Model 3 with feature selector
    qty_artifacts = joblib.load(ARTIFACTS_DIR / "model_order_qty_forecast.pkl")
    if isinstance(qty_artifacts, tuple):
        reg_qty, feature_selector = qty_artifacts
    else:
        reg_qty, feature_selector = qty_artifacts, None

    le_material = joblib.load(ARTIFACTS_DIR / "label_encoder_material.pkl")
    le_scenario = joblib.load(ARTIFACTS_DIR / "label_encoder_scenario.pkl")

    with open(ARTIFACTS_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)

    with open(ARTIFACTS_DIR / "training_metrics.json") as f:
        metrics = json.load(f)

    return clf, reg_prob, reg_qty, feature_selector, le_material, le_scenario, feature_cols, metrics


def load_and_prepare_data():
    """Load and prepare all data for predictions"""
    print("Loading data...")

    from demand_forecast_ml_pipeline import (
        load_data, merge_all, engineer_features,
        encode_categoricals, build_feature_matrix
    )

    replen, orders, scrap, material, supplier, demand_forecast, production, wc_util, machine = load_data()
    df = merge_all(replen, orders, scrap, material, supplier, demand_forecast, production, wc_util, machine)
    df = engineer_features(df)

    return df


def generate_predictions(df, clf, reg_prob, reg_qty, feature_selector, le_material, le_scenario, feature_cols):
    """Generate predictions on all data"""
    print("Generating predictions...")

    from demand_forecast_ml_pipeline import encode_categoricals, build_feature_matrix

    df_encoded, _, _ = encode_categoricals(df, fit=False, le_material=le_material, le_scenario=le_scenario)

    # Remove targets for clean X matrix
    target_cols = ["shortage_flag", "shortage_probability", "replenishment_order_qty"]
    df_for_pred = df_encoded.drop(columns=[c for c in target_cols if c in df_encoded.columns], errors='ignore')

    X, _ = build_feature_matrix(df_for_pred, feature_cols)
    X = X.fillna(X.median())

    # Generate predictions
    results = df_encoded[["week_no", "material_no"]].copy()

    # Actual values (if available)
    if "shortage_flag" in df_encoded.columns:
        results["actual_shortage_flag"] = df_encoded["shortage_flag"]
    if "shortage_probability" in df_encoded.columns:
        results["actual_shortage_prob"] = df_encoded["shortage_probability"]
    if "replenishment_order_qty" in df_encoded.columns:
        results["actual_order_qty"] = df_encoded["replenishment_order_qty"]

    # Predictions
    results["pred_shortage_flag"] = clf.predict(X)
    results["pred_shortage_confidence"] = clf.predict_proba(X)[:, 1]
    results["pred_shortage_prob"] = reg_prob.predict(X).clip(0, 1)

    X_qty = feature_selector.transform(X) if feature_selector else X
    results["pred_order_qty"] = reg_qty.predict(X_qty).clip(0).astype(int)

    # Add contextual info
    if "stock_level" in df_encoded.columns:
        results["stock_level"] = df_encoded["stock_level"]
    if "reorder_point" in df_encoded.columns:
        results["reorder_point"] = df_encoded["reorder_point"]
    if "supplier_otif_pct" in df_encoded.columns:
        results["supplier_otif"] = df_encoded["supplier_otif_pct"]

    return results


def calculate_model_accuracy(results):
    """Calculate accuracy metrics"""
    metrics_summary = {}

    if "actual_shortage_flag" in results.columns:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        acc = accuracy_score(results["actual_shortage_flag"], results["pred_shortage_flag"])
        prec = precision_score(results["actual_shortage_flag"], results["pred_shortage_flag"], zero_division=0)
        rec = recall_score(results["actual_shortage_flag"], results["pred_shortage_flag"], zero_division=0)
        f1 = f1_score(results["actual_shortage_flag"], results["pred_shortage_flag"], zero_division=0)

        metrics_summary["classifier"] = {
            "accuracy": f"{acc*100:.1f}%",
            "precision": f"{prec*100:.1f}%",
            "recall": f"{rec*100:.1f}%",
            "f1_score": f"{f1*100:.1f}%"
        }

    if "actual_shortage_prob" in results.columns:
        from sklearn.metrics import mean_absolute_error, r2_score

        mae = mean_absolute_error(results["actual_shortage_prob"], results["pred_shortage_prob"])
        r2 = r2_score(results["actual_shortage_prob"], results["pred_shortage_prob"])

        metrics_summary["probability"] = {
            "mae": f"{mae:.4f}",
            "r2": f"{r2:.4f}"
        }

    if "actual_order_qty" in results.columns:
        from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

        mae = mean_absolute_error(results["actual_order_qty"], results["pred_order_qty"])
        mape = mean_absolute_percentage_error(results["actual_order_qty"], results["pred_order_qty"]) * 100
        r2 = r2_score(results["actual_order_qty"], results["pred_order_qty"])

        metrics_summary["order_qty"] = {
            "mae": f"{mae:.2f} units",
            "mape": f"{mape:.2f}%",
            "r2": f"{r2:.4f}"
        }

    return metrics_summary


def generate_html_report(results, metrics, metrics_summary):
    """Generate HTML report"""
    print("Generating HTML report...")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Top shortages
    high_risk = results[results["pred_shortage_confidence"] > 0.7].sort_values("pred_shortage_confidence", ascending=False).head(10)

    # Model performance
    metrics_html = ""
    for model_name, model_metrics in metrics_summary.items():
        metrics_html += f"<h3>{model_name.replace('_', ' ').title()}</h3><ul>"
        for k, v in model_metrics.items():
            metrics_html += f"<li><strong>{k.replace('_', ' ').title()}:</strong> {v}</li>"
        metrics_html += "</ul>"

    # High risk materials table
    high_risk_html = high_risk[[
        "week_no", "material_no", "pred_shortage_confidence",
        "pred_shortage_prob", "pred_order_qty"
    ]].to_html(index=False, classes="table")

    # Summary stats
    total_materials = results["material_no"].nunique()
    avg_shortage_prob = results["pred_shortage_prob"].mean()
    high_risk_count = (results["pred_shortage_confidence"] > 0.7).sum()

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ML Prediction Report - {timestamp}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        .summary-box {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; padding: 10px; background: white; border-left: 4px solid #3498db; }}
        .metric strong {{ color: #2c3e50; font-size: 24px; display: block; }}
        .metric span {{ color: #7f8c8d; font-size: 12px; }}
        .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .table th {{ background: #3498db; color: white; padding: 12px; text-align: left; }}
        .table td {{ padding: 10px; border-bottom: 1px solid #ecf0f1; }}
        .table tr:hover {{ background: #f8f9fa; }}
        .alert {{ background: #e74c3c; color: white; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        ul {{ line-height: 1.8; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 ML Production Planning Prediction Report</h1>
        <p><strong>Generated:</strong> {timestamp}</p>

        <div class="summary-box">
            <h2>📊 Summary Statistics</h2>
            <div class="metric">
                <strong>{total_materials}</strong>
                <span>UNIQUE MATERIALS</span>
            </div>
            <div class="metric">
                <strong>{avg_shortage_prob:.1%}</strong>
                <span>AVG SHORTAGE PROBABILITY</span>
            </div>
            <div class="metric">
                <strong>{high_risk_count}</strong>
                <span>HIGH RISK MATERIALS (>70%)</span>
            </div>
        </div>

        <h2>🔥 Top 10 High-Risk Materials</h2>
        <div class="alert">
            <strong>⚠️ ACTION REQUIRED:</strong> {high_risk_count} materials have shortage probability > 70%
        </div>
        {high_risk_html}

        <h2>📈 Model Performance</h2>
        {metrics_html}

        <h2>💾 Export Data</h2>
        <p>Full prediction CSV saved to: <code>reports/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv</code></p>
    </div>
</body>
</html>
"""

    return html


def main():
    print("="*60)
    print("ML PREDICTION REPORT GENERATOR")
    print("="*60)

    # Load models
    clf, reg_prob, reg_qty, feature_selector, le_material, le_scenario, feature_cols, metrics = load_models()

    # Load data
    df = load_and_prepare_data()

    # Generate predictions
    results = generate_predictions(df, clf, reg_prob, reg_qty, feature_selector, le_material, le_scenario, feature_cols)

    # Calculate accuracy
    metrics_summary = calculate_model_accuracy(results)

    # Save predictions CSV
    csv_filename = REPORTS_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results.to_csv(csv_filename, index=False)
    print(f"\n✅ Predictions saved to: {csv_filename}")

    # Generate HTML report
    html = generate_html_report(results, metrics, metrics_summary)
    html_filename = REPORTS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(html_filename, "w") as f:
        f.write(html)
    print(f"✅ HTML report saved to: {html_filename}")

    print("\n" + "="*60)
    print("REPORT GENERATION COMPLETE")
    print("="*60)
    print(f"\nOpen the report in your browser: {html_filename.absolute()}")


if __name__ == "__main__":
    main()
