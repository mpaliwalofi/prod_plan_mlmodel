"""
Standalone sanity check for M-1847 example
Tests the core business logic without API dependencies
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("SANITY CHECK: M-1847, Week 1")
print("=" * 80)

# Load models and encoders
ARTIFACTS_DIR = Path("ml_artifacts")

gbr = joblib.load(ARTIFACTS_DIR / "model_order_qty_forecast.pkl")
rf_shortage = joblib.load(ARTIFACTS_DIR / "model_shortage_unified.pkl")
le_mat = joblib.load(ARTIFACTS_DIR / "label_encoder_material.pkl")
le_scen = joblib.load(ARTIFACTS_DIR / "label_encoder_scenario.pkl")

with open(ARTIFACTS_DIR / "feature_columns.json") as f:
    feat = json.load(f)
FEAT_REG = feat["regression"]
FEAT_SHORTAGE = feat["shortage"]

print(f"\nLoaded models:")
print(f"  Regression features: {len(FEAT_REG)}")
print(f"  Shortage features: {len(FEAT_SHORTAGE)}")

# Test case from bug report
print("\n" + "=" * 80)
print("TEST CASE: M-1847, Week 1")
print("=" * 80)

material_no = "M-1847"
week_no = 1
scenario = "NORMAL"
forecast_qty = 196.0
stock_level = 343.0  # CRITICAL
replenishment_order_qty = 458.0  # CRITICAL

print(f"\nInput:")
print(f"  material_no: {material_no}")
print(f"  week_no: {week_no}")
print(f"  scenario: {scenario}")
print(f"  forecast_qty: {forecast_qty}")
print(f"  stock_level: {stock_level}")
print(f"  replenishment_order_qty: {replenishment_order_qty}")

# Calculate business features
available_supply = stock_level + replenishment_order_qty
demand_gap = forecast_qty - available_supply

print(f"\nCalculated:")
print(f"  available_supply = {stock_level} + {replenishment_order_qty} = {available_supply}")
print(f"  demand_gap = {forecast_qty} - {available_supply} = {demand_gap}")

# Build feature dict with defaults
data = {
    'week_no': week_no,
    'forecast_qty': forecast_qty,
    'historical_avg_qty': 180.0,
    'seasonal_index': 1.0,
    'deviation_flag': 0,
    'confidence_score': 0.85,
    'supplier_lead_time_days': 7,
    'stock_level': stock_level,
    'reorder_point': 200.0,
    'safety_stock': 100.0,
    'replenishment_order_qty': replenishment_order_qty,
    'supplier_otif_pct': 92.74,
    'unit_cost_eur': 10.0,
    'standard_batch_qty': 50.0,
    'otif_pct': 92.74,
    'avg_lead_time_days': 7,
    'risk_score_m5': 40.0,
    'scrap_contribution_pct': 0.0,
    'avg_start_delay': 0.0,
    'avg_finish_delay': 0.0,
    'avg_overload_prob': 0.0,
    'avg_throughput_dev': 0.0,
    'n_orders': 1,
    'avg_defect_rate': 0.0,
    'avg_scrap_rate': 0.0,
    'avg_scrap_risk': 0.0,
    'total_scrap_cost': 0.0,
    'total_requested_qty': 0.0,
    'total_confirmed_qty': 0.0,
    'n_customer_orders': 0,
    'vip_order_count': 0,
    'at_risk_order_count': 0,
    'avg_wc_utilization': 70.0,
    'avg_wc_downtime': 0.0,
    'avg_wc_delay': 0.0,
    'avg_wc_overload': 0.0,
    'max_wc_utilization': 85.0
}

# Derived features
data['available_supply'] = available_supply
data['demand_gap'] = demand_gap
data['qty_deviation'] = forecast_qty - data['historical_avg_qty']
data['qty_deviation_pct'] = data['qty_deviation'] / data['historical_avg_qty']
data['stock_coverage_weeks'] = stock_level / (forecast_qty + 1e-6)
data['below_safety_stock'] = int(stock_level < data['safety_stock'])
data['below_reorder'] = int(stock_level < data['reorder_point'])
data['quality_risk_score'] = data['avg_scrap_rate'] + data['avg_defect_rate']
data['supplier_risk_flag'] = int(data['otif_pct'] < 85 or data['risk_score_m5'] > 60)
data['delay_signal'] = data['avg_start_delay'] + data['avg_finish_delay']

# Customer features
data['customer_demand_gap'] = data['total_requested_qty'] - data['total_confirmed_qty']
data['customer_confirmation_rate'] = data['total_confirmed_qty'] / (data['total_requested_qty'] + 1e-6)
data['vip_order_ratio'] = data['vip_order_count'] / (data['n_customer_orders'] + 1e-6)
data['at_risk_order_ratio'] = data['at_risk_order_count'] / (data['n_customer_orders'] + 1e-6)

# Work centre features
data['capacity_constraint_flag'] = int(data['avg_wc_utilization'] > 80)
data['high_downtime_flag'] = int(data['avg_wc_downtime'] > 0.5)

# Encode categoricals
if material_no in le_mat.classes_:
    data['material_encoded'] = le_mat.transform([material_no])[0]
else:
    data['material_encoded'] = 0
    print(f"  WARNING: {material_no} not in training data, using default encoding")

if scenario in le_scen.classes_:
    data['scenario_encoded'] = le_scen.transform([scenario])[0]
else:
    data['scenario_encoded'] = 0

# Build feature matrices
X_shortage = pd.DataFrame([data])[FEAT_SHORTAGE].fillna(0)

print(f"\nFeature matrix shape: {X_shortage.shape}")

# Predict
shortage_prob = rf_shortage.predict_proba(X_shortage)[0, 1]
ml_shortage_flag = 1 if shortage_prob > 0.5 else 0

# Apply business rules
if demand_gap > 0:
    final_flag = 1
    reason = f"Demand exceeds supply by {demand_gap:.1f} units"
else:
    final_flag = ml_shortage_flag
    if ml_shortage_flag == 1:
        reason = f"ML predicted shortage (probability: {shortage_prob:.2%})"
    else:
        reason = "Sufficient supply available"

recommended_order_qty = max(0, demand_gap)

if shortage_prob >= 0.7:
    risk_level = "High"
elif shortage_prob >= 0.4:
    risk_level = "Medium"
else:
    risk_level = "Low"

print("\n" + "=" * 80)
print("MODEL OUTPUTS:")
print("=" * 80)
print(f"  ml_shortage_probability: {shortage_prob:.4f}")
print(f"  shortage_flag: {final_flag}")
print(f"  recommended_order_qty: {recommended_order_qty:.2f}")
print(f"  risk_level: {risk_level}")
print(f"  reason: {reason}")

print("\n" + "=" * 80)
print("EXPECTED (from bug report):")
print("=" * 80)
print("  demand_gap: -605.0")
print("  shortage_flag: 0")
print("  shortage_probability: < 0.3 (low)")
print("  risk_level: Low")
print("  recommended_order_qty: 0")

print("\n" + "=" * 80)
print("VALIDATION:")
print("=" * 80)

checks = [
    ("demand_gap == -605", abs(demand_gap - (-605)) < 1.0, f"Got: {demand_gap:.1f}"),
    ("shortage_flag == 0", final_flag == 0, f"Got: {final_flag}"),
    ("shortage_prob < 0.3", shortage_prob < 0.3, f"Got: {shortage_prob:.4f}"),
    ("risk_level == 'Low'", risk_level == "Low", f"Got: {risk_level}"),
    ("recommended_order_qty == 0", recommended_order_qty == 0, f"Got: {recommended_order_qty:.2f}")
]

passed = 0
failed = 0

for name, result, details in checks:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status} {name} ({details})")
    if result:
        passed += 1
    else:
        failed += 1

print("\n" + "=" * 80)
if failed == 0:
    print(f"🎉 ALL {passed} CHECKS PASSED! The bug is FIXED!")
    print("=" * 80)
    exit(0)
else:
    print(f"❌ {failed} CHECKS FAILED, {passed} PASSED")
    print("=" * 80)
    exit(1)
