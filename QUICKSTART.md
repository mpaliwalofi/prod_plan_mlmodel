# Quick Start Guide

## What You Just Built

A complete ML pipeline with 3 trained models for demand forecasting:

1. **Shortage Flag Classifier** - Predicts if shortage will occur (0/1)
2. **Shortage Probability Regressor** - Estimates shortage risk (0.0-1.0)
3. **Replenishment Qty Forecast** - Recommends order quantity (units)

## Project Status: ✅ READY TO USE

### Files Created

```
prod_plan_mlmodel/
├── demand_forecast_ml_pipeline.py  ✅ Main ML script
├── api_server.py                   ✅ FastAPI REST API
├── requirements.txt                ✅ Dependencies
├── README.md                       ✅ Full documentation
├── QUICKSTART.md                   ✅ This file
│
├── data/                           ✅ Sample CSVs (replace with your data)
│   ├── material_replenishment.csv
│   ├── customer_order_master.csv
│   ├── scrap_quality_inspection.csv
│   ├── material_master.csv
│   └── supplier_master.csv
│
└── ml_artifacts/                   ✅ Trained models
    ├── model_shortage_classifier.pkl
    ├── model_shortage_probability.pkl
    ├── model_order_qty_forecast.pkl
    ├── label_encoder_material.pkl
    ├── label_encoder_scenario.pkl
    ├── feature_columns.json
    └── training_metrics.json
```

## Current Model Performance

Based on sample data (45 rows, 15 weeks):

| Model | Metric | Score | Interpretation |
|-------|--------|-------|----------------|
| **Model 1: Classifier** | Accuracy | 100% | Perfect on test set (may be overfit due to small data) |
| **Model 2: Probability** | MAE | 0.031 | Average error ±3% probability points |
| **Model 2: Probability** | R² | 0.76 | Explains 76% of variance - Good! |
| **Model 3: Order Qty** | MAE | 0.27 units | Low error (needs more data for better forecasts) |

⚠️ **Note:** These models are trained on sample data. Replace the CSV files with your real data and retrain for production use.

## How to Use Right Now

### 1. Test the Trained Models (CLI)

Predict on the existing sample data:

```bash
python demand_forecast_ml_pipeline.py --mode predict --input data/material_replenishment.csv
```

This creates `ml_predictions_output.csv` with ML predictions.

### 2. Start the API Server

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Then open your browser to: http://localhost:8000/docs (interactive API documentation)

### 3. Test the API

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Get Predictions:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "material_no": "MAT001",
        "week_no": 16,
        "stock_level": 300,
        "reorder_point": 200,
        "safety_stock": 100,
        "supplier_otif_pct": 95,
        "supplier_lead_time_days": 7,
        "total_ordered": 150,
        "avg_scrap_rate": 2.5,
        "standard_batch_qty": 50,
        "scenario": "RISK"
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [
    {
      "material_no": "MAT001",
      "week_no": 16,
      "ml_shortage_flag": 0,
      "ml_shortage_flag_confidence": 0.0,
      "ml_shortage_probability": 0.3783,
      "ml_forecast_qty": 68
    }
  ]
}
```

## Next Steps to Production

### Step 1: Replace Sample Data with Your Real Data

1. Copy your actual CSV files to the `data/` folder
2. Make sure they have the required columns (see README.md)
3. At minimum you need 52+ weeks of data for good time-series forecasts

### Step 2: Retrain with Your Data

```bash
python demand_forecast_ml_pipeline.py --mode train
```

This will:
- Load your real data
- Engineer features
- Train 3 new models
- Save them to `ml_artifacts/`
- Show you the actual performance metrics

### Step 3: Validate Model Performance

Check `ml_artifacts/training_metrics.json`:

- **Model 1 (Classifier):** ROC-AUC > 0.80 is good
- **Model 2 (Probability):** R² > 0.60 is acceptable, > 0.75 is good
- **Model 3 (Order Qty):** MAPE < 30% is good (needs 52+ weeks of data)

If performance is poor:
- Collect more historical data
- Check for missing/incorrect values
- Add more features (seasonality, weather, promotions)

### Step 4: Integrate with Your n8n Workflow

**Option A: Call API from n8n**

Add an HTTP Request node in your n8n workflow:

1. Node Type: HTTP Request
2. Method: POST
3. URL: `http://your-server:8000/predict`
4. Body Type: JSON
5. Body: Pass your replenishment data as `{"records": [...]}`

**Option B: Schedule Batch Predictions**

Run weekly predictions and save to CSV/database:

```bash
# In cron or n8n Schedule Trigger
python demand_forecast_ml_pipeline.py --mode predict --input this_weeks_data.csv
# Then upload ml_predictions_output.csv to Supabase
```

**Option C: Embed in LLM Prompt**

Use the ML predictions as input to your Groq LLM analysis:

```javascript
// In n8n Code node
const mlResponse = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: JSON.stringify({ records: replenishmentData })
});
const mlPreds = await mlResponse.json();

// Now use mlPreds.predictions in your LLM prompt
const prompt = `Analyze these materials with ML predictions:
Material ${mat}: Shortage Probability ${prob}, Forecast Qty ${qty}...`;
```

### Step 5: Set Up Automatic Retraining

Add a monthly retraining schedule:

```bash
# In n8n Schedule Trigger (monthly)
curl -X POST http://localhost:8000/retrain
```

Or use a cron job:
```bash
# crontab -e
0 0 1 * * cd /path/to/prod_plan_mlmodel && python demand_forecast_ml_pipeline.py --mode train
```

## Feature Importances (From Sample Data)

The top features driving shortage predictions:

1. **scenario_encoded** (69.6%) - NORMAL/RISK/CRITICAL scenario
2. **stock_roll3** (13.8%) - 3-week rolling average of stock
3. **stock_level** (10.9%) - Current inventory level
4. **demand_vs_batch** (2.9%) - Demand vs batch size ratio
5. **stock_buffer_ratio** (1.2%) - Safety margin above reorder point

This tells you: **Scenario tags and recent stock trends are the strongest predictors.**

## Troubleshooting

### Issue: "FileNotFoundError" when training

**Solution:** Make sure all 5 CSV files are in the `data/` folder:
```bash
ls data/
# Should show all 5 files
```

### Issue: API won't start

**Solution:** Make sure models are trained first:
```bash
python demand_forecast_ml_pipeline.py --mode train
```

### Issue: Low model performance

**Solutions:**
1. Add more historical data (need 52+ weeks minimum)
2. Check data quality (no missing values)
3. Add more features (see README.md for custom features)

### Issue: Unknown material/scenario in API

**Solution:** The API maps unknown values to the first known class. This is safe but may reduce accuracy. Retrain with all materials/scenarios present in your data.

## Commands Cheat Sheet

```bash
# Train models
python demand_forecast_ml_pipeline.py --mode train

# Predict on new data
python demand_forecast_ml_pipeline.py --mode predict --input newdata.csv

# Generate API template (already done)
python demand_forecast_ml_pipeline.py --mode api_template

# Start API server
uvicorn api_server:app --host 0.0.0.0 --port 8000

# Test API health
curl http://localhost:8000/health

# Get predictions
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @request.json

# Trigger retraining via API
curl -X POST http://localhost:8000/retrain
```

## What's in Each Model

### Model 1: Shortage Flag Classifier (Binary)

**Input:** Stock level, supplier OTIF, scenario, lag features, etc.
**Output:** 0 (no shortage) or 1 (shortage)
**Use:** Alert generation, binary decisions

### Model 2: Shortage Probability Regressor (Continuous)

**Input:** Same features as Model 1
**Output:** Probability score 0.0 to 1.0
**Use:** Risk prioritization, LLM analysis, dashboard heatmaps

### Model 3: Replenishment Qty Forecast (Regression)

**Input:** Same features + demand history
**Output:** Recommended order quantity in units
**Use:** Procurement planning, inventory optimization

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (is API running?) |
| `/predict` | POST | Get ML predictions for new data |
| `/metrics` | GET | View training metrics |
| `/retrain` | POST | Trigger model retraining |
| `/docs` | GET | Interactive API documentation |

## Full Documentation

See [README.md](README.md) for complete documentation including:
- Detailed feature engineering guide
- Model architecture explanations
- Advanced usage and customization
- Integration patterns
- Performance tuning

## Support

For questions or issues:
1. Check [README.md](README.md) troubleshooting section
2. Review training metrics in `ml_artifacts/training_metrics.json`
3. Inspect feature importances in training output

---

**Status:** ✅ Models trained, API tested, ready to integrate with your n8n workflow!

**Next Step:** Replace sample CSVs with your real data and retrain.
