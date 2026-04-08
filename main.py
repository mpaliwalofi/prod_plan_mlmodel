"""
api_server.py - FastAPI wrapper for ML demand forecasting models
Run: uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Dict, Optional

app = FastAPI(title="Demand Forecast ML API", version="1.0")

ARTIFACTS_DIR = Path("ml_artifacts")

# Global model variables (loaded on startup)
clf = None
reg_prob = None
reg_qty = None
le_mat = None
le_scen = None
FEATURE_COLS = None

@app.on_event("startup")
def load_models():
    """Load models at startup with error handling"""
    global clf, reg_prob, reg_qty, le_mat, le_scen, FEATURE_COLS

    # ✅ FIXED unwrap function
    def unwrap(model):
        # Keep unwrapping until we get actual model
        while isinstance(model, (tuple, list)):
            model = model[0]

        # If dict, take first value
        if isinstance(model, dict):
            model = list(model.values())[0]

        return model

    try:
        clf = unwrap(joblib.load(ARTIFACTS_DIR / "model_shortage_classifier.pkl"))
        reg_prob = unwrap(joblib.load(ARTIFACTS_DIR / "model_shortage_probability.pkl"))
        reg_qty = unwrap(joblib.load(ARTIFACTS_DIR / "model_order_qty_forecast.pkl"))

        le_mat = joblib.load(ARTIFACTS_DIR / "label_encoder_material.pkl")
        le_scen = joblib.load(ARTIFACTS_DIR / "label_encoder_scenario.pkl")

        with open(ARTIFACTS_DIR / "feature_columns.json") as f:
            FEATURE_COLS = json.load(f)

        print("clf type:", type(clf))
        print("reg_prob type:", type(reg_prob))
        print("reg_qty type:", type(reg_qty))

        print("✅ All models loaded successfully")

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise

class ReplenishmentRecord(BaseModel):
    material_no: Optional[str] = None
    week_no: Optional[int] = None
    stock_level: Optional[float] = None
    reorder_point: Optional[float] = None
    safety_stock: Optional[float] = None
    supplier_otif_pct: Optional[float] = None
    supplier_lead_time_days: Optional[float] = None
    total_ordered: Optional[float] = None
    avg_scrap_rate: Optional[float] = None
    scrap_risk_avg: Optional[float] = None
    avg_defect_rate: Optional[float] = None
    standard_batch_qty: Optional[float] = None
    scenario: Optional[str] = None
    # Add more fields as needed


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]


def preprocess(records: List[Dict]) -> tuple:
    df = pd.DataFrame(records)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    def col(name, default=0):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    stock = col("stock_level")
    reorder = col("reorder_point", 1)
    safety = col("safety_stock")
    otif = col("supplier_otif_pct", 100)
    lead = col("supplier_lead_time_days", 7)
    total_ordered = col("total_ordered")
    scrap_rate = col("avg_scrap_rate")
    batch_qty = col("standard_batch_qty", 1).replace(0, 1)

    df["stock_buffer_ratio"] = (stock - reorder) / (reorder + 1)
    df["stock_vs_safety"] = stock - safety
    df["otif_gap"] = 100 - otif
    df["lead_time_risk"] = lead * (1 - otif / 100)
    df["scrap_adj_demand"] = total_ordered * (1 + scrap_rate / 100)
    df["demand_vs_batch"] = total_ordered / batch_qty
    df["stock_roll3"] = stock
    df["demand_roll3"] = total_ordered
    df["stock_level_lag1"] = stock * 0.95
    df["stock_level_lag2"] = stock * 0.90
    df["shortage_prob_lag1"] = 0
    df["total_ordered_lag1"] = total_ordered

    # Encode categoricals
    if "material_no" in df.columns:
        known = set(le_mat.classes_)
        df["material_no"] = df["material_no"].astype(str).apply(
            lambda x: x if x in known else le_mat.classes_[0]
        )
        df["material_encoded"] = le_mat.transform(df["material_no"])

    scenario_col = next(
        (c for c in df.columns if "scenario" in c and "encoded" not in c), None
    )
    if scenario_col:
        known = set(le_scen.classes_)
        df[scenario_col] = df[scenario_col].astype(str).apply(
            lambda x: x if x in known else le_scen.classes_[0]
        )
        df["scenario_encoded"] = le_scen.transform(df[scenario_col])

    # Build feature matrix
    numeric_df = df.select_dtypes(include=["number"])
    for c in FEATURE_COLS:
        if c not in numeric_df.columns:
            numeric_df[c] = 0
    X = numeric_df[FEATURE_COLS].fillna(0)
    return df, X


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        df, X = preprocess(request.records)
        shortage_flags = clf.predict(X).tolist()
        flag_probs = clf.predict_proba(X)[:, 1].tolist()
        shortage_probs = reg_prob.predict(X).clip(0, 1).tolist()
        order_qtys = reg_qty.predict(X).clip(0).astype(int).tolist()

        preds = []
        for i, rec in enumerate(request.records):
            preds.append({
                "material_no": rec.get("material_no"),
                "week_no": rec.get("week_no"),
                "ml_shortage_flag": shortage_flags[i],
                "ml_shortage_flag_confidence": round(flag_probs[i], 4),
                "ml_shortage_probability": round(shortage_probs[i], 4),
                "ml_forecast_qty": order_qtys[i],
            })
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}


@app.get("/metrics")
def metrics():
    metrics_path = ARTIFACTS_DIR / "training_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {"error": "metrics not found"}


@app.post("/retrain")
def retrain():
    """Trigger retraining from within n8n or a cron job."""
    import subprocess
    result = subprocess.run(
        ["python", "demand_forecast_ml_pipeline.py", "--mode", "train"],
        capture_output=True, text=True
    )
    return {
        "status": "retrained",
        "stdout": result.stdout[-2000:],
        "returncode": result.returncode
    }
