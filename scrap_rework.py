"""
scrap_risk_api.py
=================
FastAPI inference endpoint for Scrap & Rework Agent (m5).

Run: uvicorn scrap_risk_api:app --host 0.0.0.0 --port 8001

This is the ONLY Python file needed - all training is done in scrap_rework.ipynb
"""
import os
import uvicorn

import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

app = FastAPI(
    title="Scrap & Rework Agent (m5) API",
    description="Real-time scrap risk scoring for production inspection lots",
    version="1.0.0"
)

# Global variables
model = None
reg_model = None  # Regression model for scrap percentage
features = None
encodings = None
MODELS_DIR = Path("models")

# Encoding maps (must match training)
SHIFT_MAP = {'Morning': 0, 'Afternoon': 1, 'Night': 2}
DEFECT_MAP = {
    'Operator Error': 0, 'Dimensional': 1, 'Material Fault': 2,
    'Surface Finish': 3, 'Machine Setup': 4
}


@app.on_event("startup")
def load_models():
    """Load ML models and features on startup."""
    global model, reg_model, features, encodings

    try:
        model_path = MODELS_DIR / "scrap_risk_m5.pkl"
        reg_model_path = MODELS_DIR / "scrap_pct_regressor.pkl"
        features_path = MODELS_DIR / "scrap_risk_m5_features.pkl"
        encodings_path = MODELS_DIR / "scrap_risk_m5_encodings.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Run scrap_rework.ipynb first!")

        # Load classifier
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        encodings = joblib.load(encodings_path)

        print("✓ Classifier loaded successfully")
        print(f"  Type: {type(model).__name__}")
        print(f"  Features: {len(features)}")

        # Load regressor (optional - for scrap percentage prediction)
        if reg_model_path.exists():
            reg_model = joblib.load(reg_model_path)
            print("✓ Regressor loaded successfully")
            print(f"  Type: {type(reg_model).__name__}")
        else:
            print("⚠ Regressor not found - scrap_pct will not be predicted")
            reg_model = None

    except Exception as e:
        print(f"Error loading model: {e}")
        raise


class InspectionPayload(BaseModel):
    """Input schema for inspection lot scoring."""

    # Identifiers
    inspection_lot: str
    production_order_no: int
    inspection_date: Optional[str] = None

    # Core scrap signals
    defect_rate_pct: float = Field(ge=0, le=100)
    scrap_rate_pct: float = Field(ge=0, le=100)
    rework_rate_pct: float = Field(ge=0, le=100)
    defect_type: str
    shift: str

    # Quantities
    inspected_qty: int = Field(gt=0)
    scrap_qty: int = Field(ge=0)
    rework_qty: int = Field(ge=0)
    scrap_cost_eur: float = Field(ge=0)

    # Machine health
    machine_id: str
    machine_name: str
    machine_type: str
    calibration_lag_days: int = Field(ge=0)
    maintenance_lag_days: int = Field(ge=0)
    calibration_overdue: int = Field(ge=0, le=1)
    maintenance_overdue: int = Field(ge=0, le=1)

    # Supplier
    supplier_name: str
    supplier_otif_pct: float = Field(ge=0, le=100)
    supplier_scrap_pct: float = Field(ge=0, le=100)
    supplier_risk_score: float = Field(ge=0, le=100)

    # Work centre
    work_centre: str
    wc_utilization: float = Field(ge=0, le=100)
    wc_overload_prob: float = Field(ge=0, le=1)
    order_overload_prob: float = Field(ge=0, le=1)
    throughput_deviation_pct: float

    # Material
    material_no: str
    material_group: str

    # Rolling features
    rolling_scrap_4w_machine_id: float
    rolling_scrap_4w_shift: float

    class Config:
        json_schema_extra = {
            "example": {
                "inspection_lot": "IL-01234",
                "production_order_no": 4704,
                "inspection_date": "2025-04-09",
                "defect_rate_pct": 12.5,
                "scrap_rate_pct": 8.2,
                "rework_rate_pct": 15.3,
                "defect_type": "Dimensional",
                "shift": "Night",
                "inspected_qty": 200,
                "scrap_qty": 16,
                "rework_qty": 31,
                "scrap_cost_eur": 850.50,
                "machine_id": "MC-02",
                "machine_name": "CNC Lathe MC-02",
                "machine_type": "CNC Lathe",
                "calibration_lag_days": 105,
                "maintenance_lag_days": 22,
                "calibration_overdue": 1,
                "maintenance_overdue": 0,
                "supplier_name": "Precision Metal Works GmbH",
                "supplier_otif_pct": 78.5,
                "supplier_scrap_pct": 14.2,
                "supplier_risk_score": 72.0,
                "work_centre": "WC-01",
                "wc_utilization": 88.5,
                "wc_overload_prob": 0.42,
                "order_overload_prob": 0.55,
                "throughput_deviation_pct": -12.3,
                "material_no": "M-6612",
                "material_group": "Metal Components",
                "rolling_scrap_4w_machine_id": 9.1,
                "rolling_scrap_4w_shift": 10.8
            }
        }


class ScoreResponse(BaseModel):
    """Response schema."""
    inspection_lot: str
    production_order_no: int
    scrap_risk_probability: float
    alert_level: str
    predicted_scrap_pct: Optional[float] = None  # New: predicted scrap percentage
    scrap_severity: Optional[str] = None  # New: severity flag
    timestamp: str

    # Context for LLM
    defect_type: str
    machine_id: str
    machine_name: str
    shift: str
    work_centre: str
    supplier_name: str
    supplier_risk_score: float
    supplier_otif_pct: float
    calibration_lag_days: int
    calibration_overdue: int
    maintenance_overdue: int
    scrap_rate_pct: float
    rework_rate_pct: float
    scrap_cost_eur: float
    wc_utilization: float
    material_no: str


def encode_row(data: dict) -> dict:
    """Encode a single row for model inference."""
    encoded = data.copy()

    # Shift encoding
    encoded['shift_encoded'] = SHIFT_MAP.get(data.get('shift', ''), 0)

    # Defect type encoding
    encoded['defect_type_encoded'] = DEFECT_MAP.get(data.get('defect_type', ''), 0)

    # Machine type encoding
    machine_type = data.get('machine_type', '')
    machine_types = encodings.get('machine_types', [])
    encoded['machine_type_encoded'] = (
        machine_types.index(machine_type) if machine_type in machine_types else 0
    )

    # Material group encoding
    material_group = data.get('material_group', '')
    material_groups = encodings.get('material_groups', [])
    encoded['material_group_encoded'] = (
        material_groups.index(material_group) if material_group in material_groups else 0
    )

    # Engineered features
    scrap_qty = data.get('scrap_qty', 0)
    rework_qty = data.get('rework_qty', 0)
    encoded['rework_share'] = rework_qty / (scrap_qty + rework_qty + 1e-6)

    scrap_cost = data.get('scrap_cost_eur', 0)
    inspected_qty = data.get('inspected_qty', 1)
    encoded['scrap_cost_per_unit'] = scrap_cost / (inspected_qty + 1e-6)

    return encoded


@app.post("/score", response_model=ScoreResponse)
def score_inspection(payload: InspectionPayload):
    """Score an inspection lot for scrap risk and predict scrap percentage."""
    try:
        # Encode features
        row = encode_row(payload.model_dump())

        # Build feature vector
        X = pd.DataFrame([row])[features].fillna(0)

        # Predict scrap risk probability (classifier)
        prob = float(model.predict_proba(X)[0, 1])

        # Determine alert level
        if prob >= 0.60:
            alert_level = 'ESCALATE'
        elif prob >= 0.40:
            alert_level = 'WARN'
        else:
            alert_level = 'OK'

        # Predict scrap percentage (regressor) if available
        predicted_scrap_pct = None
        scrap_severity = None

        if reg_model is not None:
            predicted_scrap_pct = float(reg_model.predict(X)[0])

            # Determine scrap severity
            if predicted_scrap_pct >= 10.0:
                scrap_severity = 'HIGH (Red)'
            elif predicted_scrap_pct >= 5.0:
                scrap_severity = 'MEDIUM (Yellow)'
            else:
                scrap_severity = 'LOW (Green)'

        return ScoreResponse(
            inspection_lot=payload.inspection_lot,
            production_order_no=payload.production_order_no,
            scrap_risk_probability=round(prob, 4),
            alert_level=alert_level,
            predicted_scrap_pct=round(predicted_scrap_pct, 2) if predicted_scrap_pct else None,
            scrap_severity=scrap_severity,
            timestamp=datetime.now().isoformat(),
            # Pass context for LLM
            defect_type=payload.defect_type,
            machine_id=payload.machine_id,
            machine_name=payload.machine_name,
            shift=payload.shift,
            work_centre=payload.work_centre,
            supplier_name=payload.supplier_name,
            supplier_risk_score=payload.supplier_risk_score,
            supplier_otif_pct=payload.supplier_otif_pct,
            calibration_lag_days=payload.calibration_lag_days,
            calibration_overdue=payload.calibration_overdue,
            maintenance_overdue=payload.maintenance_overdue,
            scrap_rate_pct=payload.scrap_rate_pct,
            rework_rate_pct=payload.rework_rate_pct,
            scrap_cost_eur=payload.scrap_cost_eur,
            wc_utilization=payload.wc_utilization,
            material_no=payload.material_no
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    import os
    import json

    model_path = MODELS_DIR / "scrap_risk_m5.pkl"
    metrics_path = MODELS_DIR / "scrap_risk_m5_metrics.json"

    training_timestamp = None
    if model_path.exists():
        mtime = os.path.getmtime(model_path)
        training_timestamp = datetime.fromtimestamp(mtime).isoformat()

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return {
        "status": "ok",
        "service": "Scrap & Rework Agent (m5)",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "feature_count": len(features) if features else 0,
        "training_timestamp": training_timestamp,
        "model_metrics": {
            "auc_roc": metrics.get("auc_roc"),
            "accuracy": metrics.get("accuracy"),
            "version": metrics.get("model_version")
        }
    }


@app.get("/thresholds")
def get_thresholds():
    """Return alert thresholds."""
    return {
        "alert_thresholds": {
            "OK": "< 0.40 (< 40%)",
            "WARN": "0.40 - 0.60 (40% - 60%)",
            "ESCALATE": "> 0.60 (> 60%)"
        },
        "escalation_actions": {
            "OK": "Log result only",
            "WARN": "Notify quality team",
            "ESCALATE": "Block order + escalate"
        }
    }


@app.get("/")
def root():
    """API documentation."""
    return {
        "service": "Scrap & Rework Agent (m5) API",
        "version": "1.0.0",
        "description": "Real-time scrap risk scoring",
        "model": "XGBoost Gradient Boosting Classifier",
        "endpoints": {
            "POST /score": "Score inspection lot",
            "GET /health": "Health check",
            "GET /thresholds": "View thresholds",
            "GET /": "This page"
        },
        "workflow": [
            "1. Train model in scrap_rework.ipynb",
            "2. Start API: uvicorn scrap_risk_api:app --port 8001",
            "3. Send POST requests to /score endpoint",
            "4. Route by alert_level in n8n workflow"
        ]
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # fallback for local testing
    uvicorn.run(app, host="0.0.0.0", port=port)
