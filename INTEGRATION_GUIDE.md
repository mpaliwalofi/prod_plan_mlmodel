# Scrap Prediction System - Integration Guide

## 🎯 Overview

This system provides **dual predictions** for quality inspection:
1. **Scrap Risk Classification** - Probability an inspection will have scrap issues (0-100%)
2. **Scrap Percentage Prediction** - Estimated scrap % for the batch

## 📊 System Architecture

```
┌─────────────────┐
│   n8n Workflow  │
│  (Orchestrator) │
└────────┬────────┘
         │
         ├─► Load data from Supabase
         │   • Inspection data
         │   • Machine master
         │   • Material master
         │   • Supplier master
         │   • Work centre data
         │
         ├─► Merge & prepare features
         │
         ├─► Call ML API
         │   POST /score
         │
         ├─► Route by alert_level:
         │   ├─► OK → Log only
         │   ├─► WARN → LLM analysis → Log
         │   └─► ESCALATE → LLM analysis → Block Order
         │
         └─► Save results to scrap_agent_log
```

## 🔧 Setup Instructions

### Step 1: Train Models (Notebook)

Run all cells in `scrap_rework.ipynb`:

```bash
# Open Jupyter/VS Code
jupyter notebook scrap_rework.ipynb
```

**Key cells:**
- **Cell 5**: Loads real data from `data/` folder
- **Cell 11**: Trains risk classifier
- **Cell 21**: Trains scrap % regressor
- **Cell 23**: Tests combined prediction
- **Cell 24**: Saves regressor model

**Output models:**
- `models/scrap_risk_m5.pkl` - XGBoost classifier
- `models/scrap_pct_regressor.pkl` - XGBoost regressor
- `models/scrap_risk_m5_features.pkl` - Feature list
- `models/scrap_risk_m5_encodings.pkl` - Encodings

### Step 2: Start API

```bash
python scrap_rework.py
```

The API will:
- Load both models from `models/` directory
- Start on port 8000 (or PORT env variable)
- Expose `/score` endpoint

### Step 3: Test API

```bash
python test_updated_api.py
```

Expected output:
```
Scrap Risk Probability:  0.xxxx (xx.x%)
Alert Level:             OK/WARN/ESCALATE
Predicted Scrap %:       x.xx%
Scrap Severity:          LOW/MEDIUM/HIGH
```

### Step 4: Configure n8n Workflow

Your workflow is already configured! It will:
1. Receive webhook with `inspection_lot`
2. Load all required data from Supabase
3. Call your API at `https://prod-plan-mlmodel-1.onrender.com/score`
4. Route based on `alert_level`

## 📡 API Reference

### POST /score

**Request:**
```json
{
  "inspection_lot": "INS-3001",
  "production_order_no": 4701,
  "defect_rate_pct": 3.0,
  "scrap_rate_pct": 2.65,
  "rework_rate_pct": 1.59,
  "defect_type": "Operator Error",
  "shift": "Night",
  "machine_id": "MC-09",
  "machine_type": "Robotic Assembly",
  "calibration_lag_days": 12,
  "maintenance_lag_days": 12,
  "calibration_overdue": 0,
  "maintenance_overdue": 0,
  "supplier_otif_pct": 95,
  "supplier_scrap_pct": 4.8,
  "supplier_risk_score": 28,
  "wc_utilization": 70.0,
  "material_group": "Fabricated Parts",
  ...
}
```

**Response:**
```json
{
  "inspection_lot": "INS-3001",
  "production_order_no": 4701,
  "scrap_risk_probability": 0.1234,
  "alert_level": "OK",
  "predicted_scrap_pct": 2.85,
  "scrap_severity": "LOW (Green)",
  "timestamp": "2026-04-10T...",
  "defect_type": "Operator Error",
  "machine_id": "MC-09",
  ...
}
```

### GET /health

Check if models are loaded:
```json
{
  "status": "ok",
  "model_loaded": true,
  "feature_count": 22,
  "model_metrics": {
    "auc_roc": 0.9951,
    "accuracy": 0.9700
  }
}
```

## 🚦 Alert Levels & Thresholds

| Level | Probability | Action |
|-------|-------------|--------|
| **OK** | < 40% | Log only |
| **WARN** | 40% - 60% | Notify quality team |
| **ESCALATE** | > 60% | Block order + escalate |

## 🎨 Scrap Severity Flags

| Severity | Predicted Scrap % | Color |
|----------|------------------|-------|
| **LOW** | < 5% | 🟢 Green |
| **MEDIUM** | 5% - 10% | 🟡 Yellow |
| **HIGH** | ≥ 10% | 🔴 Red |

## 📈 Model Performance

**Classifier (Risk Probability):**
- AUC-ROC: 0.9951
- Accuracy: 97.0%

**Regressor (Scrap %):**
- Run notebook to see MAE, RMSE, R²
- Example: MAE ~0.5-1%, RMSE ~1-2%

## 🔄 Retraining Workflow

When you get new data:

1. **Update data files** in `data/` folder
2. **Run notebook** `scrap_rework.ipynb`
3. **Restart API** to load new models
4. **Test** with `test_updated_api.py`

Models are automatically saved to `models/` directory.

## 🐛 Troubleshooting

### API won't start
```bash
# Check if models exist
ls models/
# Should show: scrap_risk_m5.pkl, scrap_pct_regressor.pkl
```

### Regression model not found
```bash
# Run notebook cells 21-24 to train and save regressor
# Then restart API
```

### n8n workflow gets errors
- Check API is running and accessible
- Verify Supabase credentials are valid
- Test with `test_updated_api.py` first

## 📝 n8n Workflow Integration

Your workflow expects these fields in the response:
- `scrap_risk_probability` - Used for routing
- `alert_level` - Routes to OK/WARN/ESCALATE paths
- `predicted_scrap_pct` - Additional context for LLM
- `scrap_severity` - Visual severity indicator
- All context fields (machine_id, shift, etc.) for LLM prompt

The LLM prompt in "Build LLM Prompt" node can now include:
```javascript
- Predicted Scrap %: ${d.predicted_scrap_pct}%
- Scrap Severity: ${d.scrap_severity}
```

## 🚀 Deployment Checklist

- [ ] Train models in notebook
- [ ] Verify models saved to `models/`
- [ ] Test API locally
- [ ] Deploy to Render/production
- [ ] Update n8n webhook URL
- [ ] Test end-to-end with real inspection
- [ ] Monitor scrap_agent_log table

## 📞 Support

Issues? Check:
1. Models exist in `models/` directory
2. API is running and accessible
3. n8n webhook is configured correctly
4. Supabase credentials are valid

---

**Last Updated:** 2026-04-10
**Version:** 1.0.0
