"""
Comprehensive API validation test suite
Tests all business rules from the requirements
"""

import requests
import json
from typing import List, Dict

# Base URL - change for deployed API
BASE_URL = "http://localhost:8000"


def create_test_record(
    material_no: str = "M-1192",
    week_no: int = 1,
    scenario: str = "NORMAL",
    forecast_qty: float = 200.0,
    stock_level: float = 100.0,
    replenishment_order_qty: float = 50.0,
    **kwargs
) -> Dict:
    """Helper to create test records with defaults"""
    record = {
        "material_no": material_no,
        "week_no": week_no,
        "scenario": scenario,
        "forecast_qty": forecast_qty,
        "stock_level": stock_level,
        "replenishment_order_qty": replenishment_order_qty,
        "historical_avg_qty": kwargs.get("historical_avg_qty", 180.0),
        "seasonal_index": kwargs.get("seasonal_index", 1.0),
        "deviation_flag": kwargs.get("deviation_flag", 0),
        "confidence_score": kwargs.get("confidence_score", 0.85),
        "supplier_lead_time_days": kwargs.get("supplier_lead_time_days", 7),
        "supplier_otif_pct": kwargs.get("supplier_otif_pct", 90.0),
        "reorder_point": kwargs.get("reorder_point", 100.0),
        "safety_stock": kwargs.get("safety_stock", 50.0),
        "standard_batch_qty": kwargs.get("standard_batch_qty", 50.0),
        "unit_cost_eur": kwargs.get("unit_cost_eur", 10.0),
        "otif_pct": kwargs.get("otif_pct", 90.0),
        "avg_lead_time_days": kwargs.get("avg_lead_time_days", 7),
        "risk_score_m5": kwargs.get("risk_score_m5", 40.0),
    }
    return record


def predict(records: List[Dict]) -> List[Dict]:
    """Call the prediction endpoint"""
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"records": records},
        timeout=30
    )
    response.raise_for_status()
    return response.json()["predictions"]


class TestBusinessRules:
    """Test all business logic rules from requirements"""

    def test_rule_1_no_false_shortages_healthy_stock(self):
        """
        Rule 1: No false shortages
        Records with demand_gap <= 0 should NOT return shortage_flag=1 and risk_level=High
        """
        print("\n" + "=" * 80)
        print("TEST: Rule 1 - No false shortages (healthy stock)")
        print("=" * 80)

        # Create 50 records with healthy stock (demand_gap < -100)
        records = []
        for i in range(50):
            records.append(create_test_record(
                material_no=f"M-{1000+i}",
                week_no=1,
                forecast_qty=200.0,
                stock_level=200.0 + (i * 5),  # Increasing stock
                replenishment_order_qty=150.0,  # Total supply >> demand
            ))

        results = predict(records)

        failures = []
        for idx, (record, pred) in enumerate(zip(records, results)):
            demand_gap = record["forecast_qty"] - (record["stock_level"] + record["replenishment_order_qty"])

            # All should have shortage_flag=0
            if pred["ml_shortage_flag"] != 0:
                failures.append(f"Record {idx}: Expected shortage_flag=0, got {pred['ml_shortage_flag']} (demand_gap={demand_gap:.1f})")

            # Risk level should be Low or Medium, never High with healthy stock
            if pred["risk_level"] == "High" and demand_gap < -50:
                failures.append(f"Record {idx}: Got risk_level=High with demand_gap={demand_gap:.1f}")

        print(f"\nTested {len(records)} records with healthy stock")
        print(f"Failures: {len(failures)}")
        if failures:
            for f in failures[:5]:  # Show first 5
                print(f"  - {f}")

        assert len(failures) == 0, f"{len(failures)} records failed Rule 1"

    def test_rule_2_no_missed_shortages(self):
        """
        Rule 2: No missed shortages
        Records with demand_gap > 0 must ALWAYS return shortage_flag=1
        """
        print("\n" + "=" * 80)
        print("TEST: Rule 2 - No missed shortages")
        print("=" * 80)

        # Create 50 records with genuine shortage (demand_gap > 100)
        records = []
        for i in range(50):
            records.append(create_test_record(
                material_no=f"M-{2000+i}",
                week_no=1,
                forecast_qty=500.0 + (i * 10),  # High demand
                stock_level=50.0,  # Low stock
                replenishment_order_qty=50.0,  # Low replenishment
            ))

        results = predict(records)

        failures = []
        for idx, (record, pred) in enumerate(zip(records, results)):
            demand_gap = record["forecast_qty"] - (record["stock_level"] + record["replenishment_order_qty"])

            if demand_gap > 0 and pred["ml_shortage_flag"] != 1:
                failures.append(f"Record {idx}: demand_gap={demand_gap:.1f} but shortage_flag={pred['ml_shortage_flag']}")

        print(f"\nTested {len(records)} records with genuine shortage")
        print(f"Failures: {len(failures)}")
        if failures:
            for f in failures[:5]:
                print(f"  - {f}")

        assert len(failures) == 0, f"{len(failures)} records failed Rule 2"

    def test_rule_3_probability_correlation(self):
        """
        Rule 3: Probability must correlate with gap
        - demand_gap < -500 → shortage_prob should be < 0.3
        - demand_gap > +400 → shortage_prob should be > 0.7
        """
        print("\n" + "=" * 80)
        print("TEST: Rule 3 - Probability correlation with demand_gap")
        print("=" * 80)

        # Test abundant supply (demand_gap = -600)
        abundant_records = []
        for i in range(25):
            abundant_records.append(create_test_record(
                material_no=f"M-{3000+i}",
                forecast_qty=200.0,
                stock_level=500.0,
                replenishment_order_qty=300.0,  # demand_gap = -600
            ))

        abundant_results = predict(abundant_records)

        failures_abundant = []
        for idx, pred in enumerate(abundant_results):
            if pred["ml_shortage_probability"] > 0.3:
                failures_abundant.append(f"Abundant stock record {idx}: prob={pred['ml_shortage_probability']:.4f} > 0.3")

        # Test severe shortage (demand_gap = +500)
        shortage_records = []
        for i in range(25):
            shortage_records.append(create_test_record(
                material_no=f"M-{3100+i}",
                forecast_qty=600.0,
                stock_level=50.0,
                replenishment_order_qty=50.0,  # demand_gap = +500
            ))

        shortage_results = predict(shortage_records)

        failures_shortage = []
        for idx, pred in enumerate(shortage_results):
            if pred["ml_shortage_probability"] < 0.7:
                failures_shortage.append(f"Severe shortage record {idx}: prob={pred['ml_shortage_probability']:.4f} < 0.7")

        print(f"\nAbundant stock tests: {len(failures_abundant)} failures")
        print(f"Severe shortage tests: {len(failures_shortage)} failures")

        total_failures = len(failures_abundant) + len(failures_shortage)
        assert total_failures == 0, f"{total_failures} records failed Rule 3"

    def test_rule_4_recommended_order_qty_logic(self):
        """
        Rule 4: recommended_order_qty = max(0, demand_gap)
        """
        print("\n" + "=" * 80)
        print("TEST: Rule 4 - Recommended order qty logic")
        print("=" * 80)

        test_cases = [
            # (forecast, stock, replen) -> expected_order_qty
            (200, 50, 50, 100),  # Shortage: need 100
            (200, 250, 50, 0),   # Excess: need 0
            (500, 100, 200, 200), # Shortage: need 200
            (100, 150, 0, 0),    # Excess: need 0
        ]

        records = []
        expected_orders = []
        for idx, (fcst, stk, rep, exp) in enumerate(test_cases):
            records.append(create_test_record(
                material_no=f"M-4{idx:03d}",
                forecast_qty=fcst,
                stock_level=stk,
                replenishment_order_qty=rep
            ))
            expected_orders.append(exp)

        results = predict(records)

        failures = []
        for idx, (pred, expected) in enumerate(zip(results, expected_orders)):
            actual = pred["recommended_order_qty"]
            if abs(actual - expected) > 1.0:
                failures.append(f"Record {idx}: Expected {expected}, got {actual}")

        print(f"\nTested {len(test_cases)} cases")
        print(f"Failures: {len(failures)}")
        if failures:
            for f in failures:
                print(f"  - {f}")

        assert len(failures) == 0, f"{len(failures)} records failed Rule 4"

    def test_rule_5_no_silent_defaults(self):
        """
        Rule 5: stock_level and replenishment_order_qty are REQUIRED
        Missing values should return 422 validation error, not silent default to 0
        """
        print("\n" + "=" * 80)
        print("TEST: Rule 5 - No silent defaults for critical fields")
        print("=" * 80)

        # Test missing stock_level
        record_no_stock = {
            "material_no": "M-TEST",
            "week_no": 1,
            "scenario": "NORMAL",
            "forecast_qty": 200.0,
            # stock_level missing!
            "replenishment_order_qty": 100.0
        }

        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json={"records": [record_no_stock]},
                timeout=30
            )
            if response.status_code == 200:
                raise AssertionError("Should have returned 422 for missing stock_level")
            assert response.status_code == 422, f"Expected 422, got {response.status_code}"
            print("✅ Correctly rejected missing stock_level (HTTP 422)")
        except requests.exceptions.RequestException as e:
            raise AssertionError(f"Request failed: {e}")

        # Test missing replenishment_order_qty
        record_no_replen = {
            "material_no": "M-TEST",
            "week_no": 1,
            "scenario": "NORMAL",
            "forecast_qty": 200.0,
            "stock_level": 100.0,
            # replenishment_order_qty missing!
        }

        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json={"records": [record_no_replen]},
                timeout=30
            )
            if response.status_code == 200:
                raise AssertionError("Should have returned 422 for missing replenishment_order_qty")
            assert response.status_code == 422, f"Expected 422, got {response.status_code}"
            print("✅ Correctly rejected missing replenishment_order_qty (HTTP 422)")
        except requests.exceptions.RequestException as e:
            raise AssertionError(f"Request failed: {e}")

    def test_zero_stock_zero_replen(self):
        """
        Edge case: stock_level=0, replenishment_order_qty=0
        Should return shortage_flag=1, risk_level=High
        """
        print("\n" + "=" * 80)
        print("TEST: Edge case - Zero stock and zero replenishment")
        print("=" * 80)

        records = []
        for i in range(50):
            records.append(create_test_record(
                material_no=f"M-{5000+i}",
                forecast_qty=100.0 + (i * 2),
                stock_level=0.0,
                replenishment_order_qty=0.0,
            ))

        results = predict(records)

        failures = []
        for idx, pred in enumerate(results):
            if pred["ml_shortage_flag"] != 1:
                failures.append(f"Record {idx}: Expected shortage_flag=1, got {pred['ml_shortage_flag']}")
            if pred["risk_level"] != "High":
                failures.append(f"Record {idx}: Expected risk_level=High, got {pred['risk_level']}")

        print(f"\nTested {len(records)} records with zero stock")
        print(f"Failures: {len(failures)}")
        if failures:
            for f in failures[:5]:
                print(f"  - {f}")

        assert len(failures) == 0, f"{len(failures)} records failed zero stock test"

    def test_excess_stock(self):
        """
        Edge case: stock_level >> forecast_qty (excess stock)
        Should return shortage_probability < 0.3
        """
        print("\n" + "=" * 80)
        print("TEST: Edge case - Excess stock")
        print("=" * 80)

        records = []
        for i in range(50):
            records.append(create_test_record(
                material_no=f"M-{6000+i}",
                forecast_qty=100.0,
                stock_level=500.0 + (i * 10),  # 5x-10x forecast
                replenishment_order_qty=200.0,
            ))

        results = predict(records)

        failures = []
        for idx, pred in enumerate(results):
            if pred["ml_shortage_probability"] > 0.3:
                failures.append(f"Record {idx}: prob={pred['ml_shortage_probability']:.4f} > 0.3")

        print(f"\nTested {len(records)} records with excess stock")
        print(f"Failures: {len(failures)}")
        if failures:
            for f in failures[:5]:
                print(f"  - {f}")

        assert len(failures) == 0, f"{len(failures)} records failed excess stock test"

    def test_m1847_sanity_check(self):
        """
        Test the specific M-1847 example from the bug report
        """
        print("\n" + "=" * 80)
        print("TEST: M-1847 Sanity Check (Original Bug)")
        print("=" * 80)

        record = create_test_record(
            material_no="M-1847",
            week_no=1,
            scenario="NORMAL",
            forecast_qty=196.0,
            stock_level=343.0,
            replenishment_order_qty=458.0,
            supplier_otif_pct=92.74,
            otif_pct=92.74,
        )

        results = predict([record])
        pred = results[0]

        print(f"\nInput: forecast={record['forecast_qty']}, stock={record['stock_level']}, replen={record['replenishment_order_qty']}")
        print(f"Expected: demand_gap=-605, shortage_flag=0, risk_level=Low")
        print(f"Got: shortage_flag={pred['ml_shortage_flag']}, prob={pred['ml_shortage_probability']:.4f}, risk={pred['risk_level']}")

        demand_gap = record["forecast_qty"] - (record["stock_level"] + record["replenishment_order_qty"])

        assert abs(demand_gap - (-605)) < 1.0, f"demand_gap should be -605, got {demand_gap}"
        assert pred["ml_shortage_flag"] == 0, f"shortage_flag should be 0, got {pred['ml_shortage_flag']}"
        assert pred["ml_shortage_probability"] < 0.3, f"probability should be < 0.3, got {pred['ml_shortage_probability']}"
        assert pred["risk_level"] == "Low", f"risk_level should be Low, got {pred['risk_level']}"
        assert pred["recommended_order_qty"] == 0, f"recommended_order_qty should be 0, got {pred['recommended_order_qty']}"

        print("✅ M-1847 sanity check PASSED")


def test_health_endpoint():
    """Test the health endpoint"""
    print("\n" + "=" * 80)
    print("TEST: Health endpoint")
    print("=" * 80)

    response = requests.get(f"{BASE_URL}/health", timeout=10)
    response.raise_for_status()
    data = response.json()

    print(f"\nHealth check response:")
    print(f"  Status: {data.get('status')}")
    print(f"  Version: {data.get('version')}")
    print(f"  Models loaded: {data.get('models_loaded')}")
    print(f"  Training timestamp: {data.get('training_timestamp')}")

    assert data["status"] == "ok"
    assert data["models_loaded"] is True
    assert data.get("version") is not None
    assert data.get("training_timestamp") is not None

    print("✅ Health endpoint PASSED")


if __name__ == "__main__":
    # Can be run directly without pytest
    print("Running API validation tests...")
    print("Make sure the API server is running on http://localhost:8000")
    print("")

    # Run health check first
    try:
        test_health_endpoint()
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("Is the API server running?")
        exit(1)

    # Run all business rule tests
    test_suite = TestBusinessRules()

    tests = [
        ("Rule 1: No false shortages", test_suite.test_rule_1_no_false_shortages_healthy_stock),
        ("Rule 2: No missed shortages", test_suite.test_rule_2_no_missed_shortages),
        ("Rule 3: Probability correlation", test_suite.test_rule_3_probability_correlation),
        ("Rule 4: Recommended order qty", test_suite.test_rule_4_recommended_order_qty_logic),
        ("Rule 5: No silent defaults", test_suite.test_rule_5_no_silent_defaults),
        ("Edge case: Zero stock", test_suite.test_zero_stock_zero_replen),
        ("Edge case: Excess stock", test_suite.test_excess_stock),
        ("M-1847 sanity check", test_suite.test_m1847_sanity_check),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"✅ {name} PASSED\n")
        except AssertionError as e:
            failed += 1
            print(f"❌ {name} FAILED: {e}\n")
        except Exception as e:
            failed += 1
            print(f"❌ {name} ERROR: {e}\n")

    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    exit(0 if failed == 0 else 1)
