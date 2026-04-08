"""
api_server.py
=============
FastAPI wrapper for ML demand forecasting models.
Run: uvicorn api_server:app --host 0.0.0.0 --port 8000

Models produced by prod_plan_ml_pipeline.ipynb (the source of truth for current pkl files):
  model_order_qty_forecast.pkl    → GradientBoostingRegressor (plain)
  model_shortage_classifier.pkl   → RandomForestClassifier (plain)
  model_shortage_probability.pkl  → (LogisticRegression, StandardScaler) tuple
  label_encoder_material.pkl      → LabelEncoder
  label_encoder_scenario.pkl      → LabelEncoder
  feature_columns.json            → {"regression": [...], "classifier": [...], "probability": [...]}
"""

import json
import joblib
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Dict, Optional

app = FastAPI(title="Demand Forecast ML API", version="3.0")

ARTIFACTS_DIR = Path("ml_artifacts")

# ── Global model variables ──────────────────────────────────────────────────
gbr          = None   # GradientBoostingRegressor — order qty forecast (plain)
rfc          = None   # RandomForestClassifier    — shortage flag (plain)
lr           = None   # LogisticRegression        — shortage probability
prob_scaler  = None   # StandardScaler paired with lr
le_mat       = None
le_scen      = None
FEAT_REG     = None   # 33 features for GBR
FEAT_CLS     = None   # 27 features for RFC
FEAT_PROB    = None   # 20 features for LR


@app.on_event("startup")
def load_models():
    global gbr, rfc, lr, prob_scaler, le_mat, le_scen
    global FEAT_REG, FEAT_CLS, FEAT_PROB

    try:
        # ── Order qty — plain GBR saved by notebook ──────────────────────────
        qty_artifact = joblib.load(ARTIFACTS_DIR / "model_order_qty_forecast.pkl")
        if isinstance(qty_artifact, (tuple, list)):
            # demand_forecast_ml_pipeline.py saves (model, selector) — take model only
            gbr = qty_artifact[0]
        else:
            gbr = qty_artifact

        # ── Shortage classifier — plain RandomForestClassifier ───────────────
        cls_artifact = joblib.load(ARTIFACTS_DIR / "model_shortage_classifier.pkl")
        rfc = cls_artifact[0] if isinstance(cls_artifact, (tuple, list)) else cls_artifact

        # ── Shortage probability — (LogisticRegression, StandardScaler) tuple ─
        prob_artifact = joblib.load(ARTIFACTS_DIR / "model_shortage_probability.pkl")
        if isinstance(prob_artifact, (tuple, list)) and len(prob_artifact) == 2:
            lr, prob_scaler = prob_artifact[0], prob_artifact[1]
        else:
            lr = prob_artifact
            prob_scaler = None

        # ── Label encoders ───────────────────────────────────────────────────
        le_mat  = joblib.load(ARTIFACTS_DIR / "label_encoder_material.pkl")
        le_scen = joblib.load(ARTIFACTS_DIR / "label_encoder_scenario.pkl")

        # ── Feature columns — dict with regression/classifier/probability ────
        with open(ARTIFACTS_DIR / "feature_columns.json") as f:
            feat = json.load(f)

        if isinstance(feat, dict):
            FEAT_REG  = feat["regression"]
            FEAT_CLS  = feat["classifier"]
            FEAT_PROB = feat["probability"]
        else:
            # Legacy flat list fallback
            print("WARNING: feature_columns.json is a flat list — re-run the notebook to regenerate.")
            FEAT_REG = FEAT_CLS = FEAT_PROB = feat

        print("Models loaded successfully")
        print(f"   GBR  (order qty)    : {type(gbr).__name__}")
        print(f"   RFC  (shortage flag): {type(rfc).__name__}")
        print(f"   LR   (shortage prob): {type(lr).__name__}")
        print(f"   prob_scaler loaded  : {prob_scaler is not None}")
        print(f"   FEAT_REG  : {len(FEAT_REG)} features")
        print(f"   FEAT_CLS  : {len(FEAT_CLS)} features")
        print(f"   FEAT_PROB : {len(FEAT_PROB)} features")

    except Exception as e:
        print(f"Error loading models: {e}")
        raise


# ── Schemas ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]


# ── Helpers ──────────────────────────────────────────────────────────────────

def safe_get(df: pd.DataFrame, col: str, default=0) -> pd.Series:
    return df[col] if col in df.columns else pd.Series(default, index=df.index)


def preprocess(records: List[Dict]) -> Dict[str, pd.DataFrame]:
    """
    Build three feature matrices matching the notebook's exact feature lists.

    Notebook feature sources:
      - demand_forecast CSV       : forecast_qty, historical_avg_qty, seasonal_index,
                                    deviation_flag, confidence_score
      - material_replenishment CSV: stock_level, replenishment_order_qty, supplier_otif_pct
      - production_orders (agg)   : avg_start_delay, avg_finish_delay,
                                    avg_overload_prob, avg_throughput_dev, n_orders
      - scrap_quality (agg)       : avg_defect_rate, avg_scrap_rate, avg_scrap_risk
      - material_master           : unit_cost_eur, reorder_point, safety_stock, standard_batch_qty
      - supplier_master           : otif_pct, avg_lead_time_days, risk_score_m5,
                                    supplier_lead_time_days
      - engineered                : qty_deviation, qty_deviation_pct, stock_coverage_weeks,
                                    below_safety_stock, below_reorder, quality_risk_score,
                                    supplier_risk_flag, delay_signal
    """
    df = pd.DataFrame(records)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    # ── Raw input fields ─────────────────────────────────────────────────────
    stock         = safe_get(df, "stock_level",              0)
    reorder       = safe_get(df, "reorder_point",            50)
    safety        = safe_get(df, "safety_stock",             20)
    supplier_otif = safe_get(df, "supplier_otif_pct",        95)  # from material_replenishment
    otif          = safe_get(df, "otif_pct",                 95)  # from supplier_master
    lead_sup      = safe_get(df, "supplier_lead_time_days",  5)   # from supplier_master
    lead_avg      = safe_get(df, "avg_lead_time_days",       5)   # from supplier_master
    forecast_qty  = safe_get(df, "forecast_qty",             200)
    hist_avg      = safe_get(df, "historical_avg_qty",       180)
    seasonal_idx  = safe_get(df, "seasonal_index",           1.0)
    dev_flag      = safe_get(df, "deviation_flag",           0)
    conf_score    = safe_get(df, "confidence_score",         0.8)
    replen_qty    = safe_get(df, "replenishment_order_qty",  0)
    unit_cost     = safe_get(df, "unit_cost_eur",            0)
    batch_qty     = safe_get(df, "standard_batch_qty",       50).replace(0, 1)
    risk_m5       = safe_get(df, "risk_score_m5",            0)
    avg_defect    = safe_get(df, "avg_defect_rate",          0)
    avg_scrap     = safe_get(df, "avg_scrap_rate",           0)
    avg_scrap_rsk = safe_get(df, "avg_scrap_risk",           0)
    start_delay   = safe_get(df, "avg_start_delay",          0)
    finish_delay  = safe_get(df, "avg_finish_delay",         0)
    overload_prob = safe_get(df, "avg_overload_prob",        0)
    throughput_dv = safe_get(df, "avg_throughput_dev",       0)
    n_orders      = safe_get(df, "n_orders",                 1)
    week_no       = safe_get(df, "week_no",                  1)

    # ── Write all raw fields into df ─────────────────────────────────────────
    df["stock_level"]               = stock
    df["reorder_point"]             = reorder
    df["safety_stock"]              = safety
    df["supplier_otif_pct"]         = supplier_otif
    df["otif_pct"]                  = otif
    df["supplier_lead_time_days"]   = lead_sup
    df["avg_lead_time_days"]        = lead_avg
    df["forecast_qty"]              = forecast_qty
    df["historical_avg_qty"]        = hist_avg
    df["seasonal_index"]            = seasonal_idx
    df["deviation_flag"]            = dev_flag
    df["confidence_score"]          = conf_score
    df["replenishment_order_qty"]   = replen_qty
    df["unit_cost_eur"]             = unit_cost
    df["standard_batch_qty"]        = batch_qty
    df["risk_score_m5"]             = risk_m5
    df["avg_defect_rate"]           = avg_defect
    df["avg_scrap_rate"]            = avg_scrap
    df["avg_scrap_risk"]            = avg_scrap_rsk
    df["avg_start_delay"]           = start_delay
    df["avg_finish_delay"]          = finish_delay
    df["avg_overload_prob"]         = overload_prob
    df["avg_throughput_dev"]        = throughput_dv
    df["n_orders"]                  = n_orders
    df["week_no"]                   = week_no

    # ── Engineered features — exactly mirrors notebook ───────────────────────
    df["qty_deviation"]        = forecast_qty - hist_avg
    df["qty_deviation_pct"]    = df["qty_deviation"] / (hist_avg + 1e-6)
    df["stock_coverage_weeks"] = stock / (forecast_qty + 1e-6)
    df["below_safety_stock"]   = (stock < safety).astype(int)
    df["below_reorder"]        = (stock < reorder).astype(int)
    df["quality_risk_score"]   = avg_scrap.fillna(0) + avg_defect.fillna(0)
    df["supplier_risk_flag"]   = ((otif < 85) | (risk_m5 > 60)).astype(int)
    df["delay_signal"]         = start_delay.fillna(0) + finish_delay.fillna(0)

    # ── Label encoding ───────────────────────────────────────────────────────
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

    # ── Build per-model feature matrices ─────────────────────────────────────
    def build_X(feat_cols):
        X = pd.DataFrame(index=df.index)
        for col in feat_cols:
            X[col] = df[col].values if col in df.columns else 0
        return X.fillna(0).astype(float)

    return {
        "reg":  build_X(FEAT_REG),   # GBR — 33 features
        "cls":  build_X(FEAT_CLS),   # RFC — 27 features
        "prob": build_X(FEAT_PROB),  # LR  — 20 features
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        Xs = preprocess(request.records)

        X_reg  = Xs["reg"]
        X_cls  = Xs["cls"]
        X_prob = Xs["prob"]

        # ── Order qty — GradientBoostingRegressor ────────────────────────────
        order_qtys = gbr.predict(X_reg).clip(0).astype(int).tolist()

        # ── Shortage flag — RandomForestClassifier ───────────────────────────
        shortage_flags = rfc.predict(X_cls).tolist()
        flag_probs     = rfc.predict_proba(X_cls)[:, 1].tolist()

        # ── Shortage probability — LogisticRegression + StandardScaler ───────
        if prob_scaler is not None:
            X_prob_scaled = prob_scaler.transform(X_prob)
        else:
            X_prob_scaled = X_prob
        shortage_probs = lr.predict_proba(X_prob_scaled)[:, 1].clip(0, 1).tolist()

        preds = []
        for i, rec in enumerate(request.records):
            preds.append({
                "material_no":                  rec.get("material_no"),
                "week_no":                      rec.get("week_no"),
                "ml_shortage_flag":             int(shortage_flags[i]),
                "ml_shortage_flag_confidence":  round(flag_probs[i], 4),
                "ml_shortage_probability":      round(shortage_probs[i], 4),
                "ml_forecast_qty":              order_qtys[i],
            })

        return {"predictions": preds}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": all([gbr, rfc, lr]),
        "model_types": {
            "order_qty":     type(gbr).__name__ if gbr else None,
            "shortage_flag": type(rfc).__name__ if rfc else None,
            "shortage_prob": type(lr).__name__  if lr  else None,
        },
        "prob_scaler_active": prob_scaler is not None,
        "feature_counts": {
            "regression":  len(FEAT_REG)  if FEAT_REG  else 0,
            "classifier":  len(FEAT_CLS)  if FEAT_CLS  else 0,
            "probability": len(FEAT_PROB) if FEAT_PROB else 0,
        },
    }


@app.get("/debug-features")
def debug_features():
    return {
        "regression":  {"count": len(FEAT_REG),  "columns": FEAT_REG},
        "classifier":  {"count": len(FEAT_CLS),  "columns": FEAT_CLS},
        "probability": {"count": len(FEAT_PROB), "columns": FEAT_PROB},
    }


@app.get("/metrics")
def metrics():
    metrics_path = ARTIFACTS_DIR / "training_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {"error": "metrics not found"}