"""
Sanity check for M-1847 example from the bug report
"""

import sys
import json
import joblib
import pandas as pd
from pathlib import Path

# Import the API components
from main import RecordInput, preprocess, apply_business_rules

# Load models manually
ARTIFACTS_DIR = Path("ml_artifacts")

gbr_artifact = joblib.load(ARTIFACTS_DIR / "model_order_qty_forecast.pkl")
gbr = gbr_artifact[0] if isinstance(gbr_artifact, (tuple, list)) else gbr_artifact

shortage_artifact = joblib.load(ARTIFACTS_DIR / "model_shortage_unified.pkl")
rf_shortage = shortage_artifact[0] if isinstance(shortage_artifact, (tuple, list)) else shortage_artifact

le_mat = joblib.load(ARTIFACTS_DIR / "label_encoder_material.pkl")
le_scen = joblib.load(ARTIFACTS_DIR / "label_encoder_scenario.pkl")

with open(ARTIFACTS_DIR / "feature_columns.json") as f:
    feat = json.load(f)
FEAT_REG = feat["regression"]
FEAT_SHORTAGE = feat["shortage"]

print("Models loaded successfully")

print("=" * 80)
print("SANITY CHECK: M-1847, Week 1")
print("=" * 80)

# Real example from bug report
test_record = RecordInput(
    material_no="M-1847",
    week_no=1,
    scenario="NORMAL",
    forecast_qty=196.0,
    stock_level=343.0,  # CRITICAL - was missing before
    replenishment_order_qty=458.0,  # CRITICAL - was missing before
    historical_avg_qty=180.0,
    seasonal_index=1.0,
    deviation_flag=0,
    confidence_score=0.85,
    supplier_lead_time_days=7,
    reorder_point=200.0,
    safety_stock=100.0,
    supplier_otif_pct=92.74,
    otif_pct=92.74,
    unit_cost_eur=10.0,
    standard_batch_qty=50.0,
    avg_lead_time_days=7,
    risk_score_m5=40.0
)

print("\nInput:")
print(f"  material_no: {test_record.material_no}")
print(f"  week_no: {test_record.week_no}")
print(f"  scenario: {test_record.scenario}")
print(f"  forecast_qty: {test_record.forecast_qty}")
print(f"  stock_level: {test_record.stock_level}")
print(f"  replenishment_order_qty: {test_record.replenishment_order_qty}")

# Process
Xs = preprocess([test_record])
raw_df = Xs["raw_df"]
X_shortage = Xs["shortage"]

# Get predictions
shortage_prob = rf_shortage.predict_proba(X_shortage)[0, 1]
demand_gap = raw_df["demand_gap"].iloc[0]

# Apply business rules
result = apply_business_rules(
    ml_shortage_flag=(1 if shortage_prob > 0.5 else 0),
    ml_shortage_prob=shortage_prob,
    demand_gap=demand_gap
)

print("\nCalculated Values:")
print(f"  available_supply = {test_record.stock_level} + {test_record.replenishment_order_qty} = {test_record.stock_level + test_record.replenishment_order_qty}")
print(f"  demand_gap = {test_record.forecast_qty} - {test_record.stock_level + test_record.replenishment_order_qty} = {demand_gap:.2f}")

print("\nModel Outputs:")
print(f"  ml_shortage_probability: {shortage_prob:.4f}")
print(f"  shortage_flag: {result['final_flag']}")
print(f"  recommended_order_qty: {result['recommended_order_qty']:.2f}")
print(f"  risk_level: {result['risk_level']}")
print(f"  reason: {result['reason']}")

print("\n" + "=" * 80)
print("EXPECTED VALUES (from bug report):")
print("=" * 80)
print("  demand_gap: -605.0 (196 - 801)")
print("  shortage_flag: 0 (no shortage)")
print("  risk_level: Low (abundant supply)")
print("  recommended_order_qty: 0 (no order needed)")

print("\n" + "=" * 80)
print("VALIDATION:")
print("=" * 80)

checks = []

# Check 1: demand_gap calculation
expected_gap = 196 - (343 + 458)
actual_gap = demand_gap
check1 = abs(expected_gap - actual_gap) < 1.0
checks.append(("demand_gap == -605", check1, f"Expected: {expected_gap:.1f}, Got: {actual_gap:.1f}"))

# Check 2: shortage_flag should be 0
check2 = result['final_flag'] == 0
checks.append(("shortage_flag == 0", check2, f"Expected: 0, Got: {result['final_flag']}"))

# Check 3: shortage_probability should be LOW
check3 = shortage_prob < 0.3
checks.append(("shortage_probability < 0.3", check3, f"Expected: < 0.3, Got: {shortage_prob:.4f}"))

# Check 4: risk_level should be Low
check4 = result['risk_level'] == "Low"
checks.append(("risk_level == 'Low'", check4, f"Expected: Low, Got: {result['risk_level']}"))

# Check 5: recommended_order_qty should be 0
check5 = result['recommended_order_qty'] == 0
checks.append(("recommended_order_qty == 0", check5, f"Expected: 0, Got: {result['recommended_order_qty']}"))

for name, passed, details in checks:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {name}: {details}")

all_passed = all(c[1] for c in checks)

print("\n" + "=" * 80)
if all_passed:
    print("🎉 ALL CHECKS PASSED! The bug is FIXED!")
else:
    print("❌ SOME CHECKS FAILED - Review the output above")
print("=" * 80)

sys.exit(0 if all_passed else 1)
