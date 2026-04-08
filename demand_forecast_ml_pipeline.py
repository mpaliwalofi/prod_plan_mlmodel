"""
demand_forecast_ml_pipeline.py
================================
ML Pipeline for Demand Forecasting Agent
Trains 3 models:
  Model 1 - Shortage Flag Classifier (GradientBoostingClassifier)
  Model 2 - Shortage Probability Regressor (GradientBoostingRegressor)
  Model 3 - Replenishment Order Qty Forecast (GradientBoostingRegressor)

Usage:
  python demand_forecast_ml_pipeline.py --mode train
  python demand_forecast_ml_pipeline.py --mode predict --input new_week_replenishment.csv
  python demand_forecast_ml_pipeline.py --mode api_template
"""

import os
import json
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("ml_artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PATHS = {
    "classifier": ARTIFACTS_DIR / "model_shortage_classifier.pkl",
    "probability": ARTIFACTS_DIR / "model_shortage_probability.pkl",
    "order_qty": ARTIFACTS_DIR / "model_order_qty_forecast.pkl",
    "le_material": ARTIFACTS_DIR / "label_encoder_material.pkl",
    "le_scenario": ARTIFACTS_DIR / "label_encoder_scenario.pkl",
    "feature_cols": ARTIFACTS_DIR / "feature_columns.json",
    "metrics": ARTIFACTS_DIR / "training_metrics.json",
}

# Test split: last N weeks held out
TEST_WEEKS = 3


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
def load_data():
    """Load all 9 source CSV files."""
    print("\n[1/5] Loading source data files...")

    replen = pd.read_csv(DATA_DIR / "material_replenishment.csv")
    orders = pd.read_csv(DATA_DIR / "customer_order_master.csv")
    scrap = pd.read_csv(DATA_DIR / "scrap_quality_inspection.csv")
    material = pd.read_csv(DATA_DIR / "material_master.csv")
    supplier = pd.read_csv(DATA_DIR / "supplier_master.csv")
    demand_forecast = pd.read_csv(DATA_DIR / "demand_forecast_sales_fixed.csv")
    production = pd.read_csv(DATA_DIR / "production_orders_fixed.csv")
    wc_util = pd.read_csv(DATA_DIR / "work_centre_utilization_fixed.csv")
    machine = pd.read_csv(DATA_DIR / "machine_master.csv")

    print(f"  material_replenishment      : {replen.shape}")
    print(f"  customer_order_master       : {orders.shape}")
    print(f"  scrap_quality_inspection    : {scrap.shape}")
    print(f"  material_master             : {material.shape}")
    print(f"  supplier_master             : {supplier.shape}")
    print(f"  demand_forecast_sales_fixed : {demand_forecast.shape}")
    print(f"  production_orders_fixed     : {production.shape}")
    print(f"  work_centre_utilization     : {wc_util.shape}")
    print(f"  machine_master              : {machine.shape}")

    return replen, orders, scrap, material, supplier, demand_forecast, production, wc_util, machine


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def aggregate_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Aggregate customer orders per week × material."""
    # Normalise column names
    orders.columns = orders.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    # Detect week and material columns
    week_col = next((c for c in orders.columns if "week" in c), None)
    mat_col = next((c for c in orders.columns if "material" in c), None)
    qty_col = next(
        (c for c in orders.columns if "qty" in c or "quantity" in c or "order_qty" in c),
        None,
    )
    val_col = next((c for c in orders.columns if "value" in c or "eur" in c), None)

    agg = {qty_col: "sum"} if qty_col else {}
    if val_col:
        agg[val_col] = "sum"

    # Optional flag columns
    for flag in ["vip", "at_risk", "priority"]:
        col = next((c for c in orders.columns if flag in c), None)
        if col:
            agg[col] = "sum"

    if not (week_col and mat_col and agg):
        # Fallback: return as-is
        return orders

    result = orders.groupby([week_col, mat_col]).agg(agg).reset_index()
    rename = {week_col: "week_no", mat_col: "material_no"}
    if qty_col:
        rename[qty_col] = "total_ordered"
    if val_col:
        rename[val_col] = "contract_value_eur"
    result.rename(columns=rename, inplace=True)
    return result


def aggregate_scrap(scrap: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scrap / quality data per week × material."""
    scrap.columns = scrap.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    week_col = next((c for c in scrap.columns if "week" in c), None)
    mat_col = next((c for c in scrap.columns if "material" in c), None)

    numeric_cols = scrap.select_dtypes(include=np.number).columns.tolist()
    for c in [week_col, mat_col]:
        if c and c in numeric_cols:
            numeric_cols.remove(c)

    if not (week_col and mat_col and numeric_cols):
        return scrap

    agg = {c: "mean" for c in numeric_cols}
    result = scrap.groupby([week_col, mat_col]).agg(agg).reset_index()
    result.rename(columns={week_col: "week_no", mat_col: "material_no"}, inplace=True)

    # Standardise key column names
    col_map = {}
    for c in result.columns:
        if "scrap" in c and "rate" in c and "avg_scrap_rate" not in result.columns:
            col_map[c] = "avg_scrap_rate"
        elif "defect" in c and "avg_defect_rate" not in result.columns:
            col_map[c] = "avg_defect_rate"
        elif "risk" in c and "scrap_risk_avg" not in result.columns:
            col_map[c] = "scrap_risk_avg"
    result.rename(columns=col_map, inplace=True)
    return result


def aggregate_production(production: pd.DataFrame) -> pd.DataFrame:
    """Aggregate production orders per week × material."""
    production.columns = production.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    week_col = next((c for c in production.columns if "week" in c), None)
    mat_col = next((c for c in production.columns if "material" in c), None)

    if not (week_col and mat_col):
        return production

    agg = {
        "start_delay_hrs": "mean",
        "finish_delay_hrs": "mean",
        "overload_probability": "mean",
        "throughput_deviation_pct": "mean",
        "production_order_no": "count",
    }

    # Add order status counts
    if "order_status" in production.columns:
        # Create dummy columns for status
        for status in production["order_status"].unique():
            col_name = f"status_{status.lower().replace(' ', '_')}_count"
            production[col_name] = (production["order_status"] == status).astype(int)
            agg[col_name] = "sum"

    result = production.groupby([week_col, mat_col]).agg(agg).reset_index()
    result.rename(columns={
        week_col: "week_no",
        mat_col: "material_no",
        "production_order_no": "production_order_count"
    }, inplace=True)

    return result


def aggregate_wc_utilization(wc_util: pd.DataFrame) -> pd.DataFrame:
    """Aggregate work centre utilization per week."""
    wc_util.columns = wc_util.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    week_col = next((c for c in wc_util.columns if "week" in c), None)
    wc_col = next((c for c in wc_util.columns if "work_centre" in c or "wc" in c), None)

    if not (week_col and wc_col):
        return wc_util

    agg = {
        "utilization_pct": "mean",
        "downtime_hrs": "sum",
        "delay_hrs": "sum",
        "overload_probability": "mean",
    }

    result = wc_util.groupby([week_col, wc_col]).agg(agg).reset_index()
    result.rename(columns={week_col: "week_no", wc_col: "work_centre"}, inplace=True)

    return result


def merge_all(replen, orders, scrap, material, supplier, demand_forecast, production, wc_util, machine) -> pd.DataFrame:
    """Merge all data sources on week_no + material_no."""
    print("\n[2/5] Merging data sources...")

    # Normalise column names on all frames
    for df in [replen, orders, scrap, material, supplier, demand_forecast, production, wc_util, machine]:
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    # Identify join keys in replen
    week_col = next((c for c in replen.columns if "week" in c), "week_no")
    mat_col = next((c for c in replen.columns if "material" in c), "material_no")
    replen.rename(columns={week_col: "week_no", mat_col: "material_no"}, inplace=True)

    # Aggregate transactional data
    orders_agg = aggregate_orders(orders)
    scrap_agg = aggregate_scrap(scrap)
    production_agg = aggregate_production(production)
    wc_util_agg = aggregate_wc_utilization(wc_util)

    df = replen.copy()

    # Merge aggregated orders
    if "week_no" in orders_agg.columns and "material_no" in orders_agg.columns:
        df = df.merge(orders_agg, on=["week_no", "material_no"], how="left")

    # Merge aggregated scrap
    if "week_no" in scrap_agg.columns and "material_no" in scrap_agg.columns:
        df = df.merge(scrap_agg, on=["week_no", "material_no"], how="left")

    # Merge material master (static, on material_no only)
    mat_col_m = next((c for c in material.columns if "material" in c), None)
    if mat_col_m:
        material.rename(columns={mat_col_m: "material_no"}, inplace=True)
        df = df.merge(material, on="material_no", how="left", suffixes=("", "_mat"))

    # Merge supplier master
    sup_key = next(
        (c for c in supplier.columns if "supplier" in c and "id" in c), None
    ) or next((c for c in supplier.columns if "supplier" in c), None)
    rep_sup_key = next(
        (c for c in df.columns if "supplier" in c and ("id" in c or "no" in c)), None
    )
    if sup_key and rep_sup_key:
        supplier.rename(columns={sup_key: rep_sup_key}, inplace=True)
        df = df.merge(supplier, on=rep_sup_key, how="left", suffixes=("", "_sup"))

    # Merge demand forecast (brings historical forecast data)
    df_fc_cols = [c for c in demand_forecast.columns if c in ["week_no", "material_no", "supplier_id",
                  "forecast_qty", "historical_avg_qty", "seasonal_index", "deviation_flag", "confidence_score"]]
    if "week_no" in demand_forecast.columns and "material_no" in demand_forecast.columns:
        df = df.merge(demand_forecast[df_fc_cols], on=["week_no", "material_no"],
                      how="left", suffixes=("", "_forecast"))

    # Merge production orders aggregated
    if "week_no" in production_agg.columns and "material_no" in production_agg.columns:
        df = df.merge(production_agg, on=["week_no", "material_no"], how="left")

    # Merge work centre utilization (need to map via production orders' work_centre)
    # This is more complex - we'll add avg WC metrics per week for now
    if "week_no" in wc_util_agg.columns:
        wc_weekly = wc_util_agg.groupby("week_no").agg({
            "utilization_pct": "mean",
            "downtime_hrs": "sum",
            "delay_hrs": "sum",
            "overload_probability": "mean"
        }).reset_index()
        wc_weekly.columns = ["week_no", "avg_wc_utilization_pct", "total_wc_downtime_hrs",
                             "total_wc_delay_hrs", "avg_wc_overload_prob"]
        df = df.merge(wc_weekly, on="week_no", how="left")

    # Merge machine master (static, on work_centre if available)
    # Machine capacity can indicate bottleneck risk
    if "work_centre" in machine.columns:
        machine_agg = machine.groupby("work_centre").agg({
            "capacity_units_per_hr": "sum"
        }).reset_index()
        machine_agg.columns = ["work_centre", "wc_total_capacity"]
        # This would need work_centre in df - skip for now if not present
        if "work_centre" in df.columns:
            df = df.merge(machine_agg, on="work_centre", how="left")

    print(f"  Merged shape: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all engineered features."""
    print("\n[3/5] Engineering features...")

    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    # ── Helper: safe column fetch ──
    def col(name, default=0):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    stock = col("stock_level")
    reorder = col("reorder_point", 1).replace(0, 1)
    safety = col("safety_stock")
    otif = col("supplier_otif_pct", 100)
    lead = col("supplier_lead_time_days", 7)
    total_ordered = col("total_ordered")
    scrap_rate = col("avg_scrap_rate")
    batch_qty = col("standard_batch_qty", 1).replace(0, 1)

    # ── Supply buffer ──
    df["stock_buffer_ratio"] = (stock - reorder) / np.maximum(reorder, 1)
    df["stock_vs_safety"] = stock - safety

    # ── Supplier risk ──
    df["otif_gap"] = 100 - otif
    df["lead_time_risk"] = lead * (1 - otif / 100)

    # ── Demand adjustment ──
    df["scrap_adj_demand"] = total_ordered * (1 + scrap_rate / 100)
    df["demand_vs_batch"] = total_ordered / batch_qty

    # ── Production delay risk ──
    prod_start_delay = col("start_delay_hrs")
    prod_finish_delay = col("finish_delay_hrs")
    prod_overload = col("overload_probability")
    df["total_production_delay"] = prod_start_delay + prod_finish_delay
    df["production_risk_score"] = (prod_start_delay + prod_finish_delay) * (1 + prod_overload)

    # ── Work centre bottleneck indicators ──
    wc_util = col("avg_wc_utilization_pct")
    wc_downtime = col("total_wc_downtime_hrs")
    wc_overload = col("avg_wc_overload_prob")
    df["wc_bottleneck_risk"] = (wc_util / 100) * wc_overload + (wc_downtime / 100)
    df["capacity_pressure"] = wc_util * (1 + wc_overload)

    # ── Forecast accuracy indicators ──
    forecast_qty = col("forecast_qty")
    historical_avg = col("historical_avg_qty")
    df["forecast_vs_historical"] = forecast_qty - historical_avg
    df["forecast_deviation_pct"] = np.where(
        historical_avg > 0,
        (forecast_qty - historical_avg) / historical_avg * 100,
        0
    )

    # ── Sort for lag features ──
    sort_cols = []
    if "material_no" in df.columns:
        sort_cols.append("material_no")
    if "week_no" in df.columns:
        sort_cols.append("week_no")
    if sort_cols:
        df.sort_values(sort_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)

    # ── Lag features ──
    group_col = "material_no" if "material_no" in df.columns else None

    def make_lag(series_name, lag, new_name):
        if series_name not in df.columns:
            df[new_name] = 0
            return
        if group_col:
            df[new_name] = df.groupby(group_col)[series_name].shift(lag)
        else:
            df[new_name] = df[series_name].shift(lag)

    make_lag("stock_level", 1, "stock_level_lag1")
    make_lag("stock_level", 2, "stock_level_lag2")
    make_lag("shortage_probability", 1, "shortage_prob_lag1")
    make_lag("total_ordered", 1, "total_ordered_lag1")

    # ── Rolling averages ──
    def make_roll(series_name, window, new_name):
        if series_name not in df.columns:
            df[new_name] = 0
            return
        if group_col:
            df[new_name] = (
                df.groupby(group_col)[series_name]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
        else:
            df[new_name] = df[series_name].rolling(window, min_periods=1).mean()

    make_roll("stock_level", 3, "stock_roll3")
    make_roll("total_ordered", 3, "demand_roll3")

    print(f"  Features after engineering: {df.shape[1]} columns")
    return df


def encode_categoricals(df: pd.DataFrame, fit=True, le_material=None, le_scenario=None):
    """Label-encode material_no and scenario columns."""
    df = df.copy()

    if fit:
        le_material = LabelEncoder()
        le_scenario = LabelEncoder()

    if "material_no" in df.columns:
        if fit:
            df["material_encoded"] = le_material.fit_transform(
                df["material_no"].astype(str)
            )
        else:
            known = set(le_material.classes_)
            df["material_no"] = df["material_no"].astype(str).apply(
                lambda x: x if x in known else le_material.classes_[0]
            )
            df["material_encoded"] = le_material.transform(df["material_no"])

    scenario_col = next(
        (c for c in df.columns if "scenario" in c and "encoded" not in c), None
    )
    if scenario_col:
        if fit:
            df["scenario_encoded"] = le_scenario.fit_transform(
                df[scenario_col].astype(str)
            )
        else:
            known = set(le_scenario.classes_)
            df[scenario_col] = df[scenario_col].astype(str).apply(
                lambda x: x if x in known else le_scenario.classes_[0]
            )
            df["scenario_encoded"] = le_scenario.transform(df[scenario_col])

    return df, le_material, le_scenario


def build_feature_matrix(df: pd.DataFrame, feature_cols=None):
    """Select numeric feature columns and return X matrix."""
    TARGET_COLS = ["shortage_flag", "shortage_probability", "replenishment_order_qty"]
    ID_COLS = ["week_no", "material_no", "material_id", "supplier_id", "supplier_no"]

    exclude = set(TARGET_COLS + ID_COLS)

    # Drop object columns except already-encoded
    numeric_df = df.select_dtypes(include=[np.number])
    candidates = [c for c in numeric_df.columns if c not in exclude]

    if feature_cols is not None:
        # Inference mode: align to training columns
        missing = [c for c in feature_cols if c not in numeric_df.columns]
        for m in missing:
            numeric_df[m] = 0
        candidates = feature_cols

    return numeric_df[candidates].copy(), candidates


# ─────────────────────────────────────────────
# 3. TRAIN
# ─────────────────────────────────────────────
def train():
    replen, orders, scrap, material, supplier, demand_forecast, production, wc_util, machine = load_data()
    df = merge_all(replen, orders, scrap, material, supplier, demand_forecast, production, wc_util, machine)
    df = engineer_features(df)
    df, le_material, le_scenario = encode_categoricals(df, fit=True)

    # Drop rows missing targets
    target_cols_present = [
        c
        for c in ["shortage_flag", "shortage_probability", "replenishment_order_qty"]
        if c in df.columns
    ]
    df.dropna(subset=target_cols_present, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"\n[4/5] Training on {len(df)} rows after dropna...")

    # ── Train / test split on time ──
    if "week_no" in df.columns:
        max_week = df["week_no"].max()
        cutoff = max_week - TEST_WEEKS
        train_df = df[df["week_no"] <= cutoff].copy()
        test_df = df[df["week_no"] > cutoff].copy()
    else:
        split = int(len(df) * 0.8)
        train_df = df.iloc[:split].copy()
        test_df = df.iloc[split:].copy()

    print(f"  Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    X_train, feature_cols = build_feature_matrix(train_df)
    X_test, _ = build_feature_matrix(test_df, feature_cols)

    # Fill NaN in features
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())

    metrics = {}

    # ── Model 1: Shortage Flag Classifier ──
    print("\n  Training Model 1: Shortage Flag Classifier...")
    if "shortage_flag" in train_df.columns:
        y_train_flag = train_df["shortage_flag"].astype(int)
        y_test_flag = test_df["shortage_flag"].astype(int)

        clf = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42,
        )
        clf.fit(X_train, y_train_flag)
        y_pred_flag = clf.predict(X_test)
        y_prob_flag = clf.predict_proba(X_test)[:, 1]

        report = classification_report(y_test_flag, y_pred_flag, output_dict=True)
        roc = roc_auc_score(y_test_flag, y_prob_flag)

        print(classification_report(y_test_flag, y_pred_flag))
        print(f"  ROC-AUC: {roc:.4f}")

        joblib.dump(clf, MODEL_PATHS["classifier"])
        metrics["model1_classifier"] = {
            "roc_auc": roc,
            "accuracy": report["accuracy"],
            "f1_shortage": report.get("1", {}).get("f1-score", None),
        }
    else:
        print("  WARNING: shortage_flag column not found — skipping Model 1")

    # ── Model 2: Shortage Probability Regressor ──
    print("\n  Training Model 2: Shortage Probability Regressor...")
    if "shortage_probability" in train_df.columns:
        y_train_prob = train_df["shortage_probability"].astype(float)
        y_test_prob = test_df["shortage_probability"].astype(float)

        reg_prob = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42,
        )
        reg_prob.fit(X_train, y_train_prob)
        y_pred_prob = reg_prob.predict(X_test).clip(0, 1)

        mae = mean_absolute_error(y_test_prob, y_pred_prob)
        r2 = r2_score(y_test_prob, y_pred_prob)

        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")

        joblib.dump(reg_prob, MODEL_PATHS["probability"])
        metrics["model2_probability"] = {"mae": mae, "r2": r2}
    else:
        print("  WARNING: shortage_probability column not found — skipping Model 2")

    # ── Model 3: Replenishment Order Qty Regressor (with tuning & feature selection) ──
    print("\n  Training Model 3: Replenishment Order Qty Forecast...")
    if "replenishment_order_qty" in train_df.columns:
        y_train_qty = train_df["replenishment_order_qty"].astype(float)
        y_test_qty = test_df["replenishment_order_qty"].astype(float)

        # Step 1: Feature Selection - keep only important features
        print("    Step 1/3: Selecting important features...")
        selector = SelectFromModel(
            RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            threshold="median"  # Keep top 50% features
        )
        selector.fit(X_train, y_train_qty)
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)

        selected_features = [f for f, s in zip(feature_cols, selector.get_support()) if s]
        print(f"      Selected {len(selected_features)} features from {len(feature_cols)}")

        # Step 2: Hyperparameter Tuning with GridSearchCV
        print("    Step 2/3: Tuning hyperparameters...")
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.05, 0.1],
            'max_depth': [2, 3, 4],
            'min_samples_leaf': [3, 5, 7],
            'subsample': [0.7, 0.8]
        }

        base_model = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring='r2',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train_selected, y_train_qty)

        print(f"      Best params: {grid_search.best_params_}")
        print(f"      Best CV R²: {grid_search.best_score_:.4f}")

        # Step 3: Train final model with best params
        print("    Step 3/3: Training final model...")
        reg_qty = grid_search.best_estimator_
        y_pred_qty = reg_qty.predict(X_test_selected).clip(0)

        mae = mean_absolute_error(y_test_qty, y_pred_qty)
        mape = mean_absolute_percentage_error(y_test_qty, y_pred_qty) * 100
        r2 = r2_score(y_test_qty, y_pred_qty)

        print(f"  MAE:   {mae:.2f} units")
        print(f"  MAPE:  {mape:.2f}%")
        print(f"  R²:    {r2:.4f}")

        # Save model AND feature selector
        joblib.dump((reg_qty, selector), MODEL_PATHS["order_qty"])
        metrics["model3_order_qty"] = {
            "mae": mae, "mape": mape, "r2": r2,
            "selected_features": len(selected_features),
            "best_params": grid_search.best_params_
        }
    else:
        print("  WARNING: replenishment_order_qty column not found — skipping Model 3")

    # ── Save artifacts ──
    joblib.dump(le_material, MODEL_PATHS["le_material"])
    joblib.dump(le_scenario, MODEL_PATHS["le_scenario"])
    feature_cols_dict = {
        "regression": feature_cols,      # used by GBR (order qty)
        "classifier": feature_cols,      # used by RFC (shortage flag)
        "probability": feature_cols,     # used by LR (shortage prob)
    }

    with open(MODEL_PATHS["feature_cols"], "w") as f:
        json.dump(feature_cols_dict, f, indent=2)

    with open(MODEL_PATHS["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importances (Model 1)
    if "shortage_flag" in train_df.columns:
        fi = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(
            ascending=False
        )
        print("\n  Top 10 Feature Importances (Model 1):")
        for name, imp in fi.head(10).items():
            print(f"    {name:<35} {imp*100:.1f}%")

    print("\n[5/5] All artifacts saved to ml_artifacts/")
    print("  Training complete [OK]")


# ─────────────────────────────────────────────
# 4. PREDICT
# ─────────────────────────────────────────────
def predict(input_path: str):
    """Run prediction on a new week's replenishment CSV."""
    print(f"\n[Predict] Loading input: {input_path}")

    # Load models
    clf = joblib.load(MODEL_PATHS["classifier"]) if MODEL_PATHS["classifier"].exists() else None
    reg_prob = joblib.load(MODEL_PATHS["probability"]) if MODEL_PATHS["probability"].exists() else None

    # Load Model 3 (now includes feature selector)
    if MODEL_PATHS["order_qty"].exists():
        qty_artifacts = joblib.load(MODEL_PATHS["order_qty"])
        if isinstance(qty_artifacts, tuple):
            reg_qty, feature_selector = qty_artifacts
        else:
            reg_qty, feature_selector = qty_artifacts, None
    else:
        reg_qty, feature_selector = None, None

    le_material = joblib.load(MODEL_PATHS["le_material"]) if MODEL_PATHS["le_material"].exists() else None
    le_scenario = joblib.load(MODEL_PATHS["le_scenario"]) if MODEL_PATHS["le_scenario"].exists() else None

    with open(MODEL_PATHS["feature_cols"]) as f:
        feature_cols = json.load(f)

    # Load new data
    new_df = pd.read_csv(input_path)

    # Load supporting files for merging
    try:
        orders = pd.read_csv(DATA_DIR / "customer_order_master.csv")
        scrap = pd.read_csv(DATA_DIR / "scrap_quality_inspection.csv")
        material = pd.read_csv(DATA_DIR / "material_master.csv")
        supplier = pd.read_csv(DATA_DIR / "supplier_master.csv")
        demand_forecast = pd.read_csv(DATA_DIR / "demand_forecast_sales_fixed.csv")
        production = pd.read_csv(DATA_DIR / "production_orders_fixed.csv")
        wc_util = pd.read_csv(DATA_DIR / "work_centre_utilization_fixed.csv")
        machine = pd.read_csv(DATA_DIR / "machine_master.csv")
        df = merge_all(new_df, orders, scrap, material, supplier, demand_forecast, production, wc_util, machine)
    except Exception as e:
        print(f"  Warning: Could not fully merge ({e}). Using replenishment data only.")
        df = new_df.copy()

    df = engineer_features(df)
    df, _, _ = encode_categoricals(df, fit=False, le_material=le_material, le_scenario=le_scenario)

    X, _ = build_feature_matrix(df, feature_cols)
    X = X.fillna(X.median())

    results = df[["week_no", "material_no"]].copy() if "material_no" in df.columns else df[["week_no"]].copy()

    if clf:
        results["ml_shortage_flag"] = clf.predict(X).astype(int)
        results["ml_shortage_flag_prob"] = clf.predict_proba(X)[:, 1].round(4)

    if reg_prob:
        results["ml_shortage_probability"] = reg_prob.predict(X).clip(0, 1).round(4)

    if reg_qty:
        # Apply feature selector if available
        X_for_qty = feature_selector.transform(X) if feature_selector else X
        raw_qty = reg_qty.predict(X_for_qty).clip(0)

        # Apply formula-based floor (safety buffer based on shortage probability)
        if "ml_shortage_probability" in results.columns and "demand_roll3" in df.columns:
            safety_buffer = 1.0 + results["ml_shortage_probability"].values * 0.5
            lead_days = df.get("supplier_lead_time_days", pd.Series(7, index=df.index)).values
            batch_qty = df.get("standard_batch_qty", pd.Series(1, index=df.index)).replace(0, 1).values
            formula_qty = df["demand_roll3"].values * (lead_days / 7) * safety_buffer
            formula_qty = np.maximum(formula_qty, batch_qty)
            results["ml_forecast_qty"] = np.maximum(raw_qty, formula_qty).astype(int)
        else:
            results["ml_forecast_qty"] = raw_qty.astype(int)

    output_path = "ml_predictions_output.csv"
    results.to_csv(output_path, index=False)
    print(f"\n  Predictions saved to: {output_path}")
    print(results.head(10).to_string(index=False))
    return results


# ─────────────────────────────────────────────
# 5. API TEMPLATE GENERATOR
# ─────────────────────────────────────────────
API_TEMPLATE = '''"""
api_server.py — FastAPI wrapper for ML demand forecasting models
Run: uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Dict, Optional

app = FastAPI(title="Demand Forecast ML API", version="1.0")

ARTIFACTS_DIR = Path("ml_artifacts")

# ── Load models at startup ──
clf       = joblib.load(ARTIFACTS_DIR / "model_shortage_classifier.pkl")
reg_prob  = joblib.load(ARTIFACTS_DIR / "model_shortage_probability.pkl")
reg_qty   = joblib.load(ARTIFACTS_DIR / "model_order_qty_forecast.pkl")
le_mat    = joblib.load(ARTIFACTS_DIR / "label_encoder_material.pkl")
le_scen   = joblib.load(ARTIFACTS_DIR / "label_encoder_scenario.pkl")
with open(ARTIFACTS_DIR / "feature_columns.json") as f:
    FEATURE_COLS = json.load(f)


class ReplenishmentRecord(BaseModel):
    material_no: Optional[str] = None
    week_no: Optional[int] = None
    stock_level: Optional[float] = None
    reorder_point: Optional[float] = None
    safety_stock: Optional[float] = None
    supplier_otif_pct: Optional[float] = None
    supplier_lead_time_days: Optional[float] = None
    total_ordered: Optional[float] = None
    avg_scrap_rate: Optional[float] = None
    scrap_risk_avg: Optional[float] = None
    avg_defect_rate: Optional[float] = None
    standard_batch_qty: Optional[float] = None
    scenario: Optional[str] = None
    # Add more fields as needed


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]


def preprocess(records: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    def col(name, default=0):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    stock = col("stock_level")
    reorder = col("reorder_point", 1)
    safety = col("safety_stock")
    otif = col("supplier_otif_pct", 100)
    lead = col("supplier_lead_time_days", 7)
    total_ordered = col("total_ordered")
    scrap_rate = col("avg_scrap_rate")
    batch_qty = col("standard_batch_qty", 1).replace(0, 1)

    df["stock_buffer_ratio"] = (stock - reorder) / (reorder + 1)
    df["stock_vs_safety"] = stock - safety
    df["otif_gap"] = 100 - otif
    df["lead_time_risk"] = lead * (1 - otif / 100)
    df["scrap_adj_demand"] = total_ordered * (1 + scrap_rate / 100)
    df["demand_vs_batch"] = total_ordered / batch_qty
    df["stock_roll3"] = stock
    df["demand_roll3"] = total_ordered
    df["stock_level_lag1"] = stock * 0.95
    df["stock_level_lag2"] = stock * 0.90
    df["shortage_prob_lag1"] = 0
    df["total_ordered_lag1"] = total_ordered

    # Encode categoricals
    if "material_no" in df.columns:
        known = set(le_mat.classes_)
        df["material_no"] = df["material_no"].astype(str).apply(
            lambda x: x if x in known else le_mat.classes_[0]
        )
        df["material_encoded"] = le_mat.transform(df["material_no"])

    scenario_col = next(
        (c for c in df.columns if "scenario" in c and "encoded" not in c), None
    )
    if scenario_col:
        known = set(le_scen.classes_)
        df[scenario_col] = df[scenario_col].astype(str).apply(
            lambda x: x if x in known else le_scen.classes_[0]
        )
        df["scenario_encoded"] = le_scen.transform(df[scenario_col])

    # Build feature matrix
    numeric_df = df.select_dtypes(include=["number"])
    for c in FEATURE_COLS:
        if c not in numeric_df.columns:
            numeric_df[c] = 0
    X = numeric_df[FEATURE_COLS].fillna(0)
    return df, X


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        df, X = preprocess(request.records)
        shortage_flags = clf.predict(X).tolist()
        flag_probs = clf.predict_proba(X)[:, 1].tolist()
        shortage_probs = reg_prob.predict(X).clip(0, 1).tolist()
        order_qtys = reg_qty.predict(X).clip(0).astype(int).tolist()

        preds = []
        for i, rec in enumerate(request.records):
            preds.append({
                "material_no": rec.get("material_no"),
                "week_no": rec.get("week_no"),
                "ml_shortage_flag": shortage_flags[i],
                "ml_shortage_flag_confidence": round(flag_probs[i], 4),
                "ml_shortage_probability": round(shortage_probs[i], 4),
                "ml_forecast_qty": order_qtys[i],
            })
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}


@app.get("/metrics")
def metrics():
    metrics_path = ARTIFACTS_DIR / "training_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {"error": "metrics not found"}


@app.post("/retrain")
def retrain():
    """Trigger retraining from within n8n or a cron job."""
    import subprocess
    result = subprocess.run(
        ["python", "demand_forecast_ml_pipeline.py", "--mode", "train"],
        capture_output=True, text=True
    )
    return {
        "status": "retrained",
        "stdout": result.stdout[-2000:],
        "returncode": result.returncode
    }
'''


def generate_api_template():
    with open("api_server.py", "w") as f:
        f.write(API_TEMPLATE)
    print("\n[API Template] api_server.py generated successfully!")
    print("  Run with:  uvicorn api_server:app --host 0.0.0.0 --port 8000")
    print("  Install:   pip install fastapi uvicorn")


# ─────────────────────────────────────────────
# 6. ENTRYPOINT
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Demand Forecast ML Pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "api_template"],
        required=True,
        help="Mode: train | predict | api_template",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to new week CSV for predict mode",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "predict":
        if not args.input:
            print("ERROR: --input <csv_path> is required for predict mode")
            exit(1)
        predict(args.input)

    elif args.mode == "api_template":
        generate_api_template()


if __name__ == "__main__":
    main()
