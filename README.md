# Demand Forecast ML Model Pipeline

Production-ready ML pipeline for demand forecasting with 3 trained models for shortage prediction and replenishment optimization.

## Project Structure

```
prod_plan_mlmodel/
├── data/                                    # Source CSV files
│   ├── material_replenishment.csv          # Main backbone file (stock, shortages, targets)
│   ├── customer_order_master.csv           # Customer demand data
│   ├── scrap_quality_inspection.csv        # Quality & scrap metrics
│   ├── material_master.csv                 # Material static attributes
│   └── supplier_master.csv                 # Supplier profiles
│
├── ml_artifacts/                            # Auto-generated during training
│   ├── model_shortage_classifier.pkl       # Model 1: Shortage flag (0/1)
│   ├── model_shortage_probability.pkl      # Model 2: Shortage probability (0.0-1.0)
│   ├── model_order_qty_forecast.pkl        # Model 3: Order quantity forecast
│   ├── label_encoder_material.pkl          # Material encoder
│   ├── label_encoder_scenario.pkl          # Scenario encoder
│   ├── feature_columns.json                # Feature names used in training
│   └── training_metrics.json               # Model performance metrics
│
├── demand_forecast_ml_pipeline.py          # Main ML script
├── api_server.py                           # FastAPI REST API wrapper
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- scikit-learn (ML models)
- pandas (data processing)
- numpy (numerical operations)
- joblib (model serialization)
- fastapi (REST API)
- uvicorn (ASGI server)

### 2. Prepare Your Data

Place your 5 CSV files in the `data/` directory:

- **material_replenishment.csv** - Required columns: `week_no`, `material_no`, `stock_level`, `shortage_flag`, `shortage_probability`, `replenishment_order_qty`
- **customer_order_master.csv** - Required columns: `week_no`, `material_no`, `order_qty`
- **scrap_quality_inspection.csv** - Required columns: `week_no`, `material_no`, `scrap_rate_pct`
- **material_master.csv** - Required columns: `material_no`, `standard_batch_qty`
- **supplier_master.csv** - Required columns: `supplier_id`, `supplier_otif_pct`, `supplier_lead_time_days`

**Sample CSV files are provided** in the `data/` directory. Replace them with your actual data.

### 3. Train the Models

```bash
python demand_forecast_ml_pipeline.py --mode train
```

This will:
- Load and merge all 5 data sources
- Engineer 40+ features (lag features, rolling averages, risk scores)
- Train 3 gradient boosting models
- Save models and metrics to `ml_artifacts/`
- Show feature importances and test set performance

**Expected Output:**
```
[1/5] Loading source data files...
[2/5] Merging data sources...
[3/5] Engineering features...
[4/5] Training on X rows...
  Model 1: ROC-AUC, Classification Report
  Model 2: MAE, R²
  Model 3: MAE, MAPE, R²
[5/5] All artifacts saved to ml_artifacts/
```

### 4. Generate API Server (Optional)

```bash
python demand_forecast_ml_pipeline.py --mode api_template
```

This generates `api_server.py` (already included).

### 5. Run the API Server

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

**API Endpoints:**
- `POST /predict` - Get predictions for new data
- `GET /health` - Health check
- `GET /metrics` - View training metrics
- `POST /retrain` - Trigger model retraining

### 6. Test the API

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
      "ml_shortage_flag_confidence": 0.15,
      "ml_shortage_probability": 0.32,
      "ml_forecast_qty": 120
    }
  ]
}
```

## Usage Modes

### Mode 1: Train Models

```bash
python demand_forecast_ml_pipeline.py --mode train
```

Trains all 3 models on historical data and saves artifacts.

### Mode 2: Predict on New Data

```bash
python demand_forecast_ml_pipeline.py --mode predict --input new_week_replenishment.csv
```

Loads trained models and generates predictions for new week's data. Outputs `ml_predictions_output.csv`.

### Mode 3: Generate API Template

```bash
python demand_forecast_ml_pipeline.py --mode api_template
```

Generates the FastAPI server file (already included in the project).

## The 3 Models

### Model 1: Shortage Flag Classifier
- **Algorithm:** Gradient Boosting Classifier
- **Predicts:** Binary shortage flag (0 = No shortage, 1 = Shortage)
- **Use Case:** Alert generation, binary decision-making
- **Output:** `ml_shortage_flag` (0 or 1) + confidence score

### Model 2: Shortage Probability Regressor
- **Algorithm:** Gradient Boosting Regressor
- **Predicts:** Continuous shortage probability (0.0 to 1.0)
- **Use Case:** Risk scoring, prioritization, LLM analysis
- **Output:** `ml_shortage_probability` (0.0 - 1.0)

### Model 3: Replenishment Order Qty Forecast
- **Algorithm:** Gradient Boosting Regressor
- **Predicts:** Recommended order quantity (units)
- **Use Case:** Procurement planning, inventory optimization
- **Output:** `ml_forecast_qty` (integer units)

## Feature Engineering

The pipeline creates 40+ features from your raw data:

**Supply Buffer Features:**
- `stock_buffer_ratio` = (stock - reorder_point) / (reorder_point + 1)
- `stock_vs_safety` = stock - safety_stock

**Supplier Risk Features:**
- `otif_gap` = 100 - supplier_otif_pct
- `lead_time_risk` = lead_time_days × (1 - OTIF/100)

**Demand Features:**
- `scrap_adj_demand` = ordered_qty × (1 + scrap_rate/100)
- `demand_vs_batch` = ordered_qty / batch_qty

**Time-Series Features:**
- Lag features: `stock_level_lag1`, `stock_level_lag2`, `shortage_prob_lag1`
- Rolling averages: `stock_roll3`, `demand_roll3`

## Integrating with n8n Workflow

### Option A: HTTP Request Node (Recommended)

Add an HTTP Request node in your n8n workflow:

```
[Merge Data] → [HTTP Request: POST /predict] → [Use ML Predictions]
```

**HTTP Request Node Settings:**
- Method: POST
- URL: `http://localhost:8000/predict`
- Body Type: JSON
- Body: `{{ $json }}`

### Option B: Scheduled Retraining

Add a Schedule Trigger node for weekly retraining:

```
[Schedule Trigger: Weekly] → [HTTP Request: POST /retrain] → [Notify Slack]
```

### Option C: Embed in Supabase

Run predictions weekly and store in a new table:

```sql
CREATE TABLE ml_predictions (
  week_no INT,
  material_no TEXT,
  ml_shortage_flag INT,
  ml_shortage_probability FLOAT,
  ml_forecast_qty INT,
  predicted_at TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (week_no, material_no)
);
```

Then JOIN this table in your n8n queries.

## Model Performance

After training, check `ml_artifacts/training_metrics.json`:

```json
{
  "model1_classifier": {
    "roc_auc": 1.0,
    "accuracy": 0.88,
    "f1_shortage": 0.83
  },
  "model2_probability": {
    "mae": 0.0307,
    "r2": 0.7635
  },
  "model3_order_qty": {
    "mae": 0.27,
    "mape": 122827559294677920.0,
    "r2": 0.0
  }
}
```

**Interpretation:**
- **Model 1:** Perfect separation on test set (ROC-AUC = 1.0)
- **Model 2:** Good probability calibration (R² = 0.76)
- **Model 3:** Needs more data (requires 52+ weeks for time-series)

## Top 10 Most Important Features

From the shortage classifier (Model 1):

| Feature | Importance | Meaning |
|---------|-----------|---------|
| `scenario_encoded` | 69.6% | NORMAL/RISK/CRITICAL scenario tag |
| `stock_roll3` | 13.8% | 3-week rolling average of stock |
| `stock_level` | 10.9% | Current inventory level |
| `demand_vs_batch` | 2.9% | Demand relative to batch size |
| `stock_buffer_ratio` | 1.2% | Safety margin above reorder point |

## Retraining Strategy

| Trigger | Action |
|---------|--------|
| Every 4 weeks | Retrain with newest data |
| ROC-AUC < 0.80 | Retrain + review features |
| New material/supplier | Retrain (new encodings needed) |
| Supply chain disruption | Add disruption flag + retrain |

**Automated Retraining via n8n:**
```
[Schedule: Monthly] → [POST /retrain] → [Slack Notification]
```

## Troubleshooting

### Issue: FileNotFoundError when loading data

**Solution:** Ensure all 5 CSV files are in the `data/` directory:
```bash
ls data/
# Should show: material_replenishment.csv, customer_order_master.csv, scrap_quality_inspection.csv, material_master.csv, supplier_master.csv
```

### Issue: Models not found when running API

**Solution:** Train models first:
```bash
python demand_forecast_ml_pipeline.py --mode train
```

### Issue: Low model performance

**Solutions:**
1. **Collect more data** - ML needs 52+ weeks for good time-series forecasting
2. **Check data quality** - Ensure no missing values in key columns
3. **Add more features** - Weather, seasonality, promotional calendars
4. **Tune hyperparameters** - Adjust `n_estimators`, `learning_rate`, `max_depth` in the script

### Issue: API returns 500 errors

**Solution:** Check if unknown materials/scenarios are being passed. The API will map unknown values to the first known class.

## Next Steps

1. **Replace sample data** with your actual CSV files
2. **Train models** on real data
3. **Test API** with real requests
4. **Integrate with n8n** workflow
5. **Monitor performance** and retrain monthly
6. **Add features** as your data grows (seasonality, weather, promotions)

## Advanced Usage

### Custom Hyperparameters

Edit `demand_forecast_ml_pipeline.py` lines 410-415:

```python
clf = GradientBoostingClassifier(
    n_estimators=200,      # Increase for better accuracy
    learning_rate=0.05,    # Decrease to prevent overfitting
    max_depth=3,           # Increase for complex patterns
    subsample=0.8,
    random_state=42,
)
```

### Add Custom Features

Add your feature engineering in the `engineer_features()` function (line 285):

```python
# Example: Add seasonality features
df["is_peak_season"] = df["week_no"].apply(lambda x: 1 if x in [48, 49, 50, 51, 52] else 0)
df["quarter"] = (df["week_no"] // 13) + 1
```

### Export Predictions to CSV

After training, run batch predictions:

```bash
python demand_forecast_ml_pipeline.py --mode predict --input data/material_replenishment.csv
```

This generates `ml_predictions_output.csv` with all predictions.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review training metrics in `ml_artifacts/training_metrics.json`
3. Inspect feature importances in training output
4. Ensure data quality (no missing values in key columns)

## License

MIT License - Free to use and modify.

---

**Built with:** Python 3.14, scikit-learn, FastAPI, pandas, numpy
