"""
Test script for updated API with scrap percentage prediction
"""
import requests
import json

# Sample payload (matching your n8n workflow output)
test_payload = {
    "inspection_lot": "INS-3001",
    "production_order_no": 4701,
    "inspection_date": "2026-01-08",
    "defect_rate_pct": 3.0,
    "scrap_rate_pct": 2.65,
    "rework_rate_pct": 1.59,
    "defect_type": "Operator Error",
    "shift": "Night",
    "inspected_qty": 189,
    "scrap_qty": 5,
    "rework_qty": 3,
    "scrap_cost_eur": 141.5,
    "machine_id": "MC-09",
    "machine_name": "Assembly Robot 9",
    "machine_type": "Robotic Assembly",
    "calibration_lag_days": 12,
    "maintenance_lag_days": 12,
    "calibration_overdue": 0,
    "maintenance_overdue": 0,
    "supplier_name": "Premium Steel Solutions",
    "supplier_otif_pct": 95,
    "supplier_scrap_pct": 4.8,
    "supplier_risk_score": 28,
    "work_centre": "WC-12",
    "wc_utilization": 70.0,
    "wc_overload_prob": 0.0,
    "order_overload_prob": 0.0,
    "throughput_deviation_pct": 0.0,
    "material_no": "M-4455",
    "material_group": "Fabricated Parts",
    "rolling_scrap_4w_machine_id": 2.65,
    "rolling_scrap_4w_shift": 2.65
}

print("=" * 70)
print("TESTING UPDATED API WITH SCRAP PERCENTAGE PREDICTION")
print("=" * 70)

# Test 1: Health check
print("\n1. Health Check:")
try:
    response = requests.get("http://localhost:8000/health")
    if response.status_code == 200:
        health = response.json()
        print(f"   Status: {health['status']}")
        print(f"   Model loaded: {health['model_loaded']}")
        print(f"   Features: {health['feature_count']}")
    else:
        print(f"   ERROR: {response.status_code}")
except Exception as e:
    print(f"   ERROR: Cannot connect to API - {e}")
    print("   Make sure the API is running: python scrap_rework.py")
    exit(1)

# Test 2: Score endpoint
print("\n2. Testing /score endpoint:")
print(f"   Inspection Lot: {test_payload['inspection_lot']}")
print(f"   Machine: {test_payload['machine_id']} ({test_payload['shift']} shift)")
print(f"   Actual Scrap Rate: {test_payload['scrap_rate_pct']}%")

try:
    response = requests.post(
        "http://localhost:8000/score",
        json=test_payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        result = response.json()
        print("\n   ✓ API RESPONSE:")
        print(f"   {'-' * 66}")
        print(f"   Scrap Risk Probability:  {result['scrap_risk_probability']:.4f} ({result['scrap_risk_probability']*100:.1f}%)")
        print(f"   Alert Level:             {result['alert_level']}")

        if result.get('predicted_scrap_pct') is not None:
            print(f"   Predicted Scrap %:       {result['predicted_scrap_pct']:.2f}%")
            print(f"   Scrap Severity:          {result['scrap_severity']}")
        else:
            print("   Predicted Scrap %:       NOT AVAILABLE (run notebook cells 20-23)")

        print(f"   {'-' * 66}")
        print(f"\n   Full Response:")
        print(f"   {json.dumps(result, indent=2)}")

    else:
        print(f"   ERROR {response.status_code}: {response.text}")

except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
