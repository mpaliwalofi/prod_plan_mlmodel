"""
deploy.py - Deployment Script for Production Planning ML Models
================================================================
Automates model training, validation, and API server deployment

Usage:
  python deploy.py --mode full       # Train + validate + start API
  python deploy.py --mode train      # Train models only
  python deploy.py --mode validate   # Validate existing models
  python deploy.py --mode api        # Start API server only
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime

# Configuration
ARTIFACTS_DIR = Path("ml_artifacts")
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

def log(message):
    """Print and log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)

    log_file = LOGS_DIR / f"deployment_{datetime.now().strftime('%Y%m%d')}.log"
    with open(log_file, "a") as f:
        f.write(log_message + "\n")


def check_data_files():
    """Verify all required CSV files exist"""
    log("Checking data files...")
    required_files = [
        "data/material_replenishment.csv",
        "data/customer_order_master.csv",
        "data/scrap_quality_inspection.csv",
        "data/material_master.csv",
        "data/supplier_master.csv",
        "data/demand_forecast_sales_fixed.csv",
        "data/production_orders_fixed.csv",
        "data/work_centre_utilization_fixed.csv",
        "data/machine_master.csv"
    ]

    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        log(f"❌ ERROR: Missing data files:")
        for f in missing:
            log(f"  - {f}")
        return False

    log(f"✅ All {len(required_files)} data files found")
    return True


def train_models():
    """Train all ML models"""
    log("="*60)
    log("STEP 1: Training ML Models")
    log("="*60)

    if not check_data_files():
        return False

    log("Starting training pipeline...")
    result = subprocess.run(
        [sys.executable, "demand_forecast_ml_pipeline.py", "--mode", "train"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        log("❌ Training failed!")
        log(result.stderr)
        return False

    log("✅ Training completed successfully")
    return True


def validate_models():
    """Validate trained models exist and check metrics"""
    log("="*60)
    log("STEP 2: Validating Models")
    log("="*60)

    required_artifacts = [
        "ml_artifacts/model_shortage_classifier.pkl",
        "ml_artifacts/model_shortage_probability.pkl",
        "ml_artifacts/model_order_qty_forecast.pkl",
        "ml_artifacts/label_encoder_material.pkl",
        "ml_artifacts/label_encoder_scenario.pkl",
        "ml_artifacts/feature_columns.json",
        "ml_artifacts/training_metrics.json"
    ]

    missing = [f for f in required_artifacts if not Path(f).exists()]
    if missing:
        log(f"❌ Missing artifacts:")
        for f in missing:
            log(f"  - {f}")
        return False

    log(f"✅ All {len(required_artifacts)} artifacts found")

    # Check metrics
    metrics_path = ARTIFACTS_DIR / "training_metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    log("\n📊 Model Performance:")
    log("-" * 60)

    # Model 1
    if "model1_classifier" in metrics:
        m1 = metrics["model1_classifier"]
        log(f"Model 1 (Shortage Classifier):")
        log(f"  ROC-AUC: {m1.get('roc_auc', 'N/A')}")
        log(f"  Status: {'✅ EXCELLENT' if m1.get('roc_auc', 0) > 0.95 else '⚠️ NEEDS REVIEW'}")

    # Model 2
    if "model2_probability" in metrics:
        m2 = metrics["model2_probability"]
        log(f"\nModel 2 (Shortage Probability):")
        log(f"  MAE: {m2.get('mae', 'N/A')}")
        log(f"  R²: {m2.get('r2', 'N/A')}")
        log(f"  Status: {'✅ GOOD' if m2.get('r2', 0) > 0.6 else '⚠️ NEEDS TUNING'}")

    # Model 3
    if "model3_order_qty" in metrics:
        m3 = metrics["model3_order_qty"]
        log(f"\nModel 3 (Order Qty Forecast):")
        log(f"  MAE: {m3.get('mae', 'N/A')} units")
        log(f"  R²: {m3.get('r2', 'N/A')}")
        if "best_params" in m3:
            log(f"  Best Params: {m3['best_params']}")
        log(f"  Status: {'✅ GOOD' if m3.get('r2', 0) > 0.5 else '⚠️ ACCEPTABLE' if m3.get('r2', 0) > 0 else '🔴 POOR'}")

    log("-" * 60)
    return True


def start_api_server():
    """Start the FastAPI server"""
    log("="*60)
    log("STEP 3: Starting API Server")
    log("="*60)

    if not Path("api_server.py").exists():
        log("⚠️ api_server.py not found, generating template...")
        result = subprocess.run(
            [sys.executable, "demand_forecast_ml_pipeline.py", "--mode", "api_template"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            log("❌ Failed to generate API template")
            return False

    log("Starting uvicorn server on http://0.0.0.0:8000")
    log("Press Ctrl+C to stop the server")
    log("API Endpoints:")
    log("  - POST /predict     : Get predictions")
    log("  - GET  /health      : Health check")
    log("  - GET  /metrics     : Training metrics")
    log("  - POST /retrain     : Trigger retraining")
    log("")

    subprocess.run(
        [sys.executable, "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"],
        check=False
    )


def main():
    parser = argparse.ArgumentParser(description="Deploy ML Production Planning System")
    parser.add_argument(
        "--mode",
        choices=["full", "train", "validate", "api"],
        required=True,
        help="Deployment mode"
    )
    args = parser.parse_args()

    log("🚀 Starting Deployment Process")
    log(f"Mode: {args.mode.upper()}")

    if args.mode == "train":
        success = train_models()
        sys.exit(0 if success else 1)

    elif args.mode == "validate":
        success = validate_models()
        sys.exit(0 if success else 1)

    elif args.mode == "api":
        start_api_server()

    elif args.mode == "full":
        # Full deployment: train -> validate -> start API
        if not train_models():
            log("❌ Deployment failed at training stage")
            sys.exit(1)

        if not validate_models():
            log("⚠️ Validation warnings detected, but proceeding...")

        log("\n" + "="*60)
        log("🎉 Deployment Complete!")
        log("="*60)
        log("\nStarting API server...")

        start_api_server()


if __name__ == "__main__":
    main()
