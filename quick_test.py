"""Quick test of the API with M-1847 case"""
import requests
import json

url = "http://localhost:8000/predict"

# M-1847 test case
payload = {
    "records": [
        {
            "material_no": "M-1847",
            "week_no": 1,
            "scenario": "NORMAL",
            "forecast_qty": 196.0,
            "stock_level": 343.0,
            "replenishment_order_qty": 458.0,
            "historical_avg_qty": 180.0,
            "seasonal_index": 1.0,
            "deviation_flag": 0,
            "confidence_score": 0.85,
            "supplier_lead_time_days": 7,
            "supplier_otif_pct": 92.74,
            "reorder_point": 200.0,
            "safety_stock": 100.0,
            "otif_pct": 92.74,
            "risk_score_m5": 40.0
        }
    ]
}

print("Sending request to API...")
print(f"Input: forecast_qty={payload['records'][0]['forecast_qty']}, stock={payload['records'][0]['stock_level']}, replen={payload['records'][0]['replenishment_order_qty']}")

response = requests.post(url, json=payload)
print(f"\nStatus: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    pred = result["predictions"][0]

    print("\nResult:")
    print(json.dumps(pred, indent=2))

    # Validation
    print("\n" + "=" * 80)
    print("VALIDATION:")
    print("=" * 80)

    expected_gap = 196 - (343 + 458)
    checks = [
        ("shortage_flag == 0", pred["ml_shortage_flag"] == 0),
        ("shortage_prob < 0.3", pred["ml_shortage_probability"] < 0.3),
        ("risk_level == Low", pred["risk_level"] == "Low"),
        ("recommended_order_qty == 0", pred["recommended_order_qty"] == 0),
        ("demand_gap == -605", abs(pred["demand_gap"] - expected_gap) < 1.0)
    ]

    passed = sum(1 for _, result in checks if result)
    for name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {name}")

    if passed == len(checks):
        print(f"\n🎉 All {len(checks)} checks PASSED!")
    else:
        print(f"\n❌ {len(checks) - passed} checks FAILED")
else:
    print(f"Error: {response.text}")
