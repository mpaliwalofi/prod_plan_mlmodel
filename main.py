"""
improved_api_server.py
======================
Improved FastAPI wrapper for ML demand forecasting models.

Key Improvements:
- Uses unified calibrated Random Forest model for shortage prediction
- Implements rule-based override: if demand_gap > 0, shortage_flag = 1
- Derives shortage_flag from probability (no contradictions)
- Returns business-friendly outputs: recommended_order_qty, risk_level, reason
- Clean JSON output format

Run: uvicorn improved_api_server:app --host 0.0.0.0 --port 8000

Models:
  model_order_qty_forecast.pkl  → GradientBoostingRegressor
  model_shortage_unified.pkl    → Calibrated RandomForestClassifier
  label_encoder_material.pkl    → LabelEncoder
  label_encoder_scenario.pkl    → LabelEncoder
  feature_columns.json          → {"regression": [...], "shortage": [...]}
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

app = FastAPI(title="Improved Demand Forecast ML API", version="4.0")

ARTIFACTS_DIR = Path("ml_artifacts")

# Global model variables
gbr = None  # GradientBoostingRegressor for forecast qty
rf_shortage = None  # Calibrated RandomForest for shortage
le_mat = None
le_scen = None
FEAT_REG = None
FEAT_SHORTAGE = None


@app.on_event("startup")
def load_models():
    global gbr, rf_shortage, le_mat, le_scen, FEAT_REG, FEAT_SHORTAGE

    try:
        # Load regression model
        gbr_artifact = joblib.load(ARTIFACTS_DIR / "model_order_qty_forecast.pkl")
        gbr = gbr_artifact[0] if isinstance(gbr_artifact, (tuple, list)) else gbr_artifact

        # Load unified shortage model (calibrated)
        shortage_artifact = joblib.load(ARTIFACTS_DIR / "model_shortage_unified.pkl")
        rf_shortage = shortage_artifact[0] if isinstance(shortage_artifact, (tuple, list)) else shortage_artifact

        # Load label encoders
        le_mat = joblib.load(ARTIFACTS_DIR / "label_encoder_material.pkl")
        le_scen = joblib.load(ARTIFACTS_DIR / "label_encoder_scenario.pkl")

        # Load feature columns
        with open(ARTIFACTS_DIR / "feature_columns.json") as f:
            feat = json.load(f)

        FEAT_REG = feat["regression"]
        FEAT_SHORTAGE = feat["shortage"]

        print("✅ Models loaded successfully")
        print(f"   GBR (order qty): {type(gbr).__name__}")
        print(f"   RF Shortage: {type(rf_shortage).__name__}")
        print(f"   FEAT_REG: {len(FEAT_REG)} features")
        print(f"   FEAT_SHORTAGE: {len(FEAT_SHORTAGE)} features")

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise


# Schemas
class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]


# Helper functions
def safe_get(df: pd.DataFrame, col: str, default=0) -> pd.Series:
    """Get column from DataFrame or return default."""
    return df[col] if col in df.columns else pd.Series(default, index=df.index)


def preprocess(records: List[Dict]) -> Dict[str, pd.DataFrame]:
    """
    Build feature matrices for regression and shortage models.
    Includes critical business feature: demand_gap
    """
    df = pd.DataFrame(records)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    # Extract raw fields
    stock = safe_get(df, "stock_level", 0)
    reorder = safe_get(df, "reorder_point", 50)
    safety = safe_get(df, "safety_stock", 20)
    supplier_otif = safe_get(df, "supplier_otif_pct", 95)
    otif = safe_get(df, "otif_pct", 95)
    lead_sup = safe_get(df, "supplier_lead_time_days", 5)
    lead_avg = safe_get(df, "avg_lead_time_days", 5)
    forecast_qty = safe_get(df, "forecast_qty", 200)
    hist_avg = safe_get(df, "historical_avg_qty", 180)
    seasonal_idx = safe_get(df, "seasonal_index", 1.0)
    dev_flag = safe_get(df, "deviation_flag", 0)
    conf_score = safe_get(df, "confidence_score", 0.8)
    replen_qty = safe_get(df, "replenishment_order_qty", 0)
    unit_cost = safe_get(df, "unit_cost_eur", 0)
    batch_qty = safe_get(df, "standard_batch_qty", 50).replace(0, 1)
    risk_m5 = safe_get(df, "risk_score_m5", 0)
    avg_defect = safe_get(df, "avg_defect_rate", 0)
    avg_scrap = safe_get(df, "avg_scrap_rate", 0)
    avg_scrap_rsk = safe_get(df, "avg_scrap_risk", 0)
    start_delay = safe_get(df, "avg_start_delay", 0)
    finish_delay = safe_get(df, "avg_finish_delay", 0)
    overload_prob = safe_get(df, "avg_overload_prob", 0)
    throughput_dv = safe_get(df, "avg_throughput_dev", 0)
    n_orders = safe_get(df, "n_orders", 1)
    week_no = safe_get(df, "week_no", 1)

    # Assign to DataFrame
    df["stock_level"] = stock
    df["reorder_point"] = reorder
    df["safety_stock"] = safety
    df["supplier_otif_pct"] = supplier_otif
    df["otif_pct"] = otif
    df["supplier_lead_time_days"] = lead_sup
    df["avg_lead_time_days"] = lead_avg
    df["forecast_qty"] = forecast_qty
    df["historical_avg_qty"] = hist_avg
    df["seasonal_index"] = seasonal_idx
    df["deviation_flag"] = dev_flag
    df["confidence_score"] = conf_score
    df["replenishment_order_qty"] = replen_qty
    df["unit_cost_eur"] = unit_cost
    df["standard_batch_qty"] = batch_qty
    df["risk_score_m5"] = risk_m5
    df["avg_defect_rate"] = avg_defect
    df["avg_scrap_rate"] = avg_scrap
    df["avg_scrap_risk"] = avg_scrap_rsk
    df["avg_start_delay"] = start_delay
    df["avg_finish_delay"] = finish_delay
    df["avg_overload_prob"] = overload_prob
    df["avg_throughput_dev"] = throughput_dv
    df["n_orders"] = n_orders
    df["week_no"] = week_no

    # CRITICAL BUSINESS FEATURE: demand_gap
    df["available_supply"] = stock + replen_qty
    df["demand_gap"] = forecast_qty - df["available_supply"]

    # Other engineered features
    df["qty_deviation"] = forecast_qty - hist_avg
    df["qty_deviation_pct"] = df["qty_deviation"] / (hist_avg + 1e-6)
    df["stock_coverage_weeks"] = stock / (forecast_qty + 1e-6)
    df["below_safety_stock"] = (stock < safety).astype(int)
    df["below_reorder"] = (stock < reorder).astype(int)
    df["quality_risk_score"] = avg_scrap.fillna(0) + avg_defect.fillna(0)
    df["supplier_risk_flag"] = ((otif < 85) | (risk_m5 > 60)).astype(int)
    df["delay_signal"] = start_delay.fillna(0) + finish_delay.fillna(0)

    # Label encoding
    if "material_no" in df.columns:
        known = set(le_mat.classes_)
        df["material_no"] = df["material_no"].astype(str).apply(
            lambda x: x if x in known else le_mat.classes_[0]
        )
        df["material_encoded"] = le_mat.transform(df["material_no"])
    else:
        df["material_encoded"] = 0

    if "scenario" in df.columns:
        known = set(le_scen.classes_)
        df["scenario"] = df["scenario"].astype(str).apply(
            lambda x: x if x in known else le_scen.classes_[0]
        )
        df["scenario_encoded"] = le_scen.transform(df["scenario"])
    else:
        df["scenario_encoded"] = 0

    # Build feature matrices
    def build_X(feat_cols):
        X = pd.DataFrame(index=df.index)
        for col in feat_cols:
            X[col] = df[col].values if col in df.columns else 0
        return X.fillna(0).astype(float)

    return {
        "reg": build_X(FEAT_REG),
        "shortage": build_X(FEAT_SHORTAGE),
        "raw_df": df  # Return for demand_gap extraction
    }


def apply_business_rules(ml_shortage_flag, ml_shortage_prob, demand_gap):
    """
    Apply business rules to ML predictions:
    1. If demand_gap > 0, force shortage_flag = 1
    2. Calculate recommended order qty
    3. Determine risk level
    4. Generate reason
    """
    # Rule-based override
    if demand_gap > 0:
        final_flag = 1
        reason = f"Demand exceeds supply by {demand_gap:.1f} units"
    else:
        final_flag = ml_shortage_flag
        if ml_shortage_flag == 1:
            reason = f"ML predicted shortage (probability: {ml_shortage_prob:.2%})"
        else:
            reason = "Sufficient supply available"

    # Recommended order quantity
    recommended_order_qty = max(0, demand_gap)

    # Risk level
    if ml_shortage_prob >= 0.7:
        risk_level = "High"
    elif ml_shortage_prob >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "final_flag": int(final_flag),
        "recommended_order_qty": round(recommended_order_qty, 2),
        "risk_level": risk_level,
        "reason": reason
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Main prediction endpoint with improved business logic.

    Returns:
        - ml_forecast_qty: Predicted demand
        - ml_shortage_probability: Calibrated probability [0-1]
        - ml_shortage_flag: Binary flag (1 if prob > 0.5 OR demand_gap > 0)
        - demand_gap: Demand - available supply
        - recommended_order_qty: How much to order (max(0, demand_gap))
        - risk_level: Low/Medium/High
        - reason: Explanation for the prediction
    """
    try:
        Xs = preprocess(request.records)
        X_reg = Xs["reg"]
        X_shortage = Xs["shortage"]
        raw_df = Xs["raw_df"]

        # 1. Forecast quantity
        forecast_qtys = gbr.predict(X_reg).clip(0).astype(float).tolist()

        # 2. Shortage probability (calibrated)
        shortage_probs = rf_shortage.predict_proba(X_shortage)[:, 1].clip(0, 1).tolist()

        # 3. Derive flag from probability
        shortage_flags = [1 if p > 0.5 else 0 for p in shortage_probs]

        # 4. Extract demand_gap
        demand_gaps = raw_df["demand_gap"].tolist()

        # Build predictions
        preds = []
        for i, rec in enumerate(request.records):
            ml_flag = shortage_flags[i]
            ml_prob = shortage_probs[i]
            demand_gap = demand_gaps[i]

            # Apply business rules
            business_output = apply_business_rules(ml_flag, ml_prob, demand_gap)

            preds.append({
                "material_no": rec.get("material_no"),
                "week_no": rec.get("week_no"),
                "scenario": rec.get("scenario"),
                "ml_forecast_qty": round(forecast_qtys[i], 2),
                "ml_shortage_probability": round(ml_prob, 4),
                "ml_shortage_flag": business_output["final_flag"],
                "demand_gap": round(demand_gap, 2),
                "recommended_order_qty": business_output["recommended_order_qty"],
                "risk_level": business_output["risk_level"],
                "reason": business_output["reason"]
            })

        return {"predictions": preds}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models_loaded": all([gbr, rf_shortage]),
        "model_types": {
            "order_qty": type(gbr).__name__ if gbr else None,
            "shortage": type(rf_shortage).__name__ if rf_shortage else None,
        },
        "feature_counts": {
            "regression": len(FEAT_REG) if FEAT_REG else 0,
            "shortage": len(FEAT_SHORTAGE) if FEAT_SHORTAGE else 0,
        },
    }


@app.get("/debug-features")
def debug_features():
    """Return feature lists for debugging."""
    return {
        "regression": {"count": len(FEAT_REG), "columns": FEAT_REG},
        "shortage": {"count": len(FEAT_SHORTAGE), "columns": FEAT_SHORTAGE},
    }


@app.get("/metrics")
def metrics():
    """Return training metrics."""
    metrics_path = ARTIFACTS_DIR / "training_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {"error": "metrics not found"}


@app.get("/")
def root():
    """Root endpoint with API documentation."""
    return {
        "title": "Improved Demand Forecast ML API",
        "version": "4.0",
        "description": "Predicts demand and shortage risk with business logic",
        "improvements": [
            "Fixed inconsistency: shortage_flag derived from probability",
            "Added demand_gap business feature",
            "Rule-based override: if demand_gap > 0, force shortage",
            "Replaced Logistic Regression with calibrated Random Forest",
            "Added business outputs: recommended_order_qty, risk_level, reason"
        ],
        "endpoints": {
            "POST /predict": "Get predictions for records",
            "GET /health": "Health check",
            "GET /metrics": "Training metrics",
            "GET /debug-features": "Feature lists"
        },
        "example_request": {
            "records": [
                {
                    "material_no": "M-1192",
                    "week_no": 5,
                    "scenario": "NORMAL",
                    "forecast_qty": 150,
                    "stock_level": 50,
                    "replenishment_order_qty": 80,
                    "historical_avg_qty": 140,
                    "seasonal_index": 1.05,
                    "confidence_score": 0.85,
                    "supplier_lead_time_days": 7,
                    "reorder_point": 100,
                    "safety_stock": 30,
                    "otif_pct": 88,
                    "risk_score_m5": 45
                }
            ]
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
