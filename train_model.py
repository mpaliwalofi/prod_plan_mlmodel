"""
Production Planning ML Model Training Script
Converted from prod_plan_model.ipynb
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import json
import os

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, roc_auc_score, accuracy_score
)

# Configuration
pd.set_option('display.max_columns', 50)

DATA_DIR = 'data/'
OUTPUT_DIR = 'ml_artifacts/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('✅ All libraries loaded successfully\n')

# ============= 1. DATA LOADING =============
print('=' * 80)
print('STEP 1: Loading data files')
print('=' * 80)

demand_forecast = pd.read_csv(DATA_DIR + 'demand_forecast_sales_fixed.csv')
material_replen = pd.read_csv(DATA_DIR + 'material_replenishment.csv')
production_orders = pd.read_csv(DATA_DIR + 'production_orders_fixed.csv')
scrap_quality = pd.read_csv(DATA_DIR + 'scrap_quality_inspection.csv')
material_master = pd.read_csv(DATA_DIR + 'material_master.csv')
supplier_master = pd.read_csv(DATA_DIR + 'supplier_master.csv')
customer_orders = pd.read_csv(DATA_DIR + 'customer_order_master.csv')
work_centre_util = pd.read_csv(DATA_DIR + 'work_centre_utilization_fixed.csv')

print(f'Demand forecast: {demand_forecast.shape}')
print(f'Material replenishment: {material_replen.shape}')
print(f'Production orders: {production_orders.shape}')
print(f'Scrap quality: {scrap_quality.shape}')
print(f'Material master: {material_master.shape}')
print(f'Supplier master: {supplier_master.shape}')
print(f'Customer orders: {customer_orders.shape}')
print(f'Work centre utilization: {work_centre_util.shape}')

# ============= 2. FEATURE ENGINEERING =============
print('\n' + '=' * 80)
print('STEP 2: Merging tables and engineering features')
print('=' * 80)

df = demand_forecast.copy()

# Merge material replenishment
df = df.merge(
    material_replen[['week_no', 'material_no', 'scenario', 'stock_level',
                     'shortage_flag', 'shortage_probability', 'replenishment_order_qty',
                     'supplier_otif_pct']],
    on=['week_no', 'material_no', 'scenario'],
    how='left',
    suffixes=('', '_replen')
)

# Aggregate production orders by material + week
prod_agg = production_orders.groupby(['week_no', 'material_no', 'scenario']).agg(
    avg_start_delay=('start_delay_hrs', 'mean'),
    avg_finish_delay=('finish_delay_hrs', 'mean'),
    avg_overload_prob=('overload_probability', 'mean'),
    avg_throughput_dev=('throughput_deviation_pct', 'mean'),
    n_orders=('production_order_no', 'count')
).reset_index()
df = df.merge(prod_agg, on=['week_no', 'material_no', 'scenario'], how='left')

# Aggregate scrap quality by material + week
scrap_agg = scrap_quality.groupby(['week_no', 'material_no', 'scenario']).agg(
    avg_defect_rate=('defect_rate_pct', 'mean'),
    avg_scrap_rate=('scrap_rate_pct', 'mean'),
    avg_scrap_risk=('scrap_risk_probability', 'mean'),
    total_scrap_cost=('scrap_cost_eur', 'sum')
).reset_index()
df = df.merge(scrap_agg, on=['week_no', 'material_no', 'scenario'], how='left')

# Aggregate customer orders by material + week
customer_agg = customer_orders.groupby(['week_no', 'material_no', 'scenario']).agg(
    total_requested_qty=('requested_qty', 'sum'),
    total_confirmed_qty=('confirmed_qty', 'sum'),
    n_customer_orders=('customer_order_no', 'count'),
    vip_order_count=('vip_flag', lambda x: (x == 'Y').sum()),
    at_risk_order_count=('delivery_status', lambda x: (x == 'At Risk').sum())
).reset_index()
df = df.merge(customer_agg, on=['week_no', 'material_no', 'scenario'], how='left')

# Aggregate work centre utilization by week + scenario
wc_agg = work_centre_util.groupby(['week_no', 'scenario']).agg(
    avg_wc_utilization=('utilization_pct', 'mean'),
    avg_wc_downtime=('downtime_hrs', 'mean'),
    avg_wc_delay=('delay_hrs', 'mean'),
    avg_wc_overload=('overload_probability', 'mean'),
    max_wc_utilization=('utilization_pct', 'max')
).reset_index()
df = df.merge(wc_agg, on=['week_no', 'scenario'], how='left')

# Merge material master
df = df.merge(
    material_master[['material_no', 'unit_cost_eur', 'reorder_point', 'safety_stock', 'standard_batch_qty']],
    on='material_no', how='left'
)

# Merge supplier master
df = df.merge(
    supplier_master[['supplier_id', 'otif_pct', 'avg_lead_time_days', 'risk_score_m5', 'scrap_contribution_pct']],
    on='supplier_id', how='left'
)

print(f'✅ All tables merged successfully')
print(f'Final merged dataframe shape: {df.shape}')

# Create engineered features
print('\nCreating engineered features...')

# CRITICAL BUSINESS FEATURE
df['demand_gap'] = df['forecast_qty'] - (df['stock_level'] + df['replenishment_order_qty'])
df['available_supply'] = df['stock_level'] + df['replenishment_order_qty']

# Basic derived features
df['qty_deviation'] = df['forecast_qty'] - df['historical_avg_qty']
df['qty_deviation_pct'] = df['qty_deviation'] / (df['historical_avg_qty'] + 1e-6)
df['stock_coverage_weeks'] = df['stock_level'] / (df['forecast_qty'] + 1e-6)
df['below_safety_stock'] = (df['stock_level'] < df['safety_stock']).astype(int)
df['below_reorder'] = (df['stock_level'] < df['reorder_point']).astype(int)
df['quality_risk_score'] = df['avg_scrap_rate'].fillna(0) + df['avg_defect_rate'].fillna(0)
df['supplier_risk_flag'] = ((df['otif_pct'] < 85) | (df['risk_score_m5'] > 60)).astype(int)
df['delay_signal'] = df['avg_start_delay'].fillna(0) + df['avg_finish_delay'].fillna(0)

# Customer order features
df['customer_demand_gap'] = df['total_requested_qty'].fillna(0) - df['total_confirmed_qty'].fillna(0)
df['customer_confirmation_rate'] = df['total_confirmed_qty'].fillna(0) / (df['total_requested_qty'].fillna(0) + 1e-6)
df['vip_order_ratio'] = df['vip_order_count'].fillna(0) / (df['n_customer_orders'].fillna(0) + 1e-6)
df['at_risk_order_ratio'] = df['at_risk_order_count'].fillna(0) / (df['n_customer_orders'].fillna(0) + 1e-6)

# Work centre capacity features
df['capacity_constraint_flag'] = (df['avg_wc_utilization'].fillna(0) > 80).astype(int)
df['high_downtime_flag'] = (df['avg_wc_downtime'].fillna(0) > 0.5).astype(int)

# Label encoding
le_material = LabelEncoder()
le_scenario = LabelEncoder()
df['material_encoded'] = le_material.fit_transform(df['material_no'])
df['scenario_encoded'] = le_scenario.fit_transform(df['scenario'])

print(f'✅ Feature engineering complete')
print(f'\nKey feature - demand_gap statistics:')
print(df['demand_gap'].describe())
print(f'\nRecords with positive demand_gap (shortage expected): {(df["demand_gap"] > 0).sum()}')

# Save label encoders
with open(OUTPUT_DIR + 'label_encoder_material.pkl', 'wb') as f:
    pickle.dump(le_material, f)
with open(OUTPUT_DIR + 'label_encoder_scenario.pkl', 'wb') as f:
    pickle.dump(le_scenario, f)
print('\n✅ Label encoders saved')

# ============= 3. MODEL 1: FORECAST QUANTITY =============
print('\n' + '=' * 80)
print('STEP 3: Training Forecast Quantity Model (GradientBoostingRegressor)')
print('=' * 80)

FEATURE_COLS_REG = [
    'week_no', 'material_encoded', 'scenario_encoded',
    'historical_avg_qty', 'seasonal_index', 'deviation_flag',
    'confidence_score', 'supplier_lead_time_days',
    'stock_level', 'reorder_point', 'replenishment_order_qty',
    'avg_start_delay', 'avg_finish_delay', 'avg_overload_prob',
    'avg_throughput_dev', 'n_orders',
    'avg_defect_rate', 'avg_scrap_rate', 'avg_scrap_risk',
    'unit_cost_eur', 'safety_stock', 'standard_batch_qty',
    'otif_pct', 'avg_lead_time_days', 'risk_score_m5',
    'qty_deviation', 'qty_deviation_pct', 'stock_coverage_weeks',
    'below_safety_stock', 'below_reorder',
    'quality_risk_score', 'supplier_risk_flag', 'delay_signal',
    'demand_gap',
    # NEW features
    'total_requested_qty', 'total_confirmed_qty', 'n_customer_orders',
    'vip_order_count', 'at_risk_order_count',
    'customer_demand_gap', 'customer_confirmation_rate',
    'vip_order_ratio', 'at_risk_order_ratio',
    'avg_wc_utilization', 'avg_wc_downtime', 'avg_wc_delay',
    'avg_wc_overload', 'max_wc_utilization',
    'capacity_constraint_flag', 'high_downtime_flag'
]

TARGET_REG = 'forecast_qty'

df_model = df[FEATURE_COLS_REG + [TARGET_REG]].dropna()
print(f'Regression model dataset shape: {df_model.shape}')

X = df_model[FEATURE_COLS_REG]
y = df_model[TARGET_REG]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.85,
    min_samples_leaf=3,
    random_state=42
)
print('Training GradientBoostingRegressor...')
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print('\n📊 Forecast Quantity Model — Test Metrics')
print(f'   MAE  : {mae:.2f}')
print(f'   RMSE : {rmse:.2f}')
print(f'   R²   : {r2:.4f}')

cv_r2 = cross_val_score(gbr, X, y, cv=5, scoring='r2')
print(f'   CV R² (5-fold): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}')

# Save model
with open(OUTPUT_DIR + 'model_order_qty_forecast.pkl', 'wb') as f:
    pickle.dump(gbr, f)
print('\n✅ model_order_qty_forecast.pkl saved')

# ============= 4. MODEL 2: SHORTAGE PREDICTION =============
print('\n' + '=' * 80)
print('STEP 4: Training Shortage Prediction Model (Calibrated Random Forest)')
print('=' * 80)

FEATURE_COLS_SHORTAGE = [
    'week_no', 'material_encoded', 'scenario_encoded',
    'forecast_qty', 'historical_avg_qty', 'seasonal_index',
    'confidence_score', 'supplier_lead_time_days',
    'stock_level', 'reorder_point', 'replenishment_order_qty',
    'available_supply',
    'demand_gap',  # CRITICAL
    'supplier_otif_pct', 'avg_start_delay', 'avg_overload_prob',
    'avg_throughput_dev', 'avg_scrap_risk', 'quality_risk_score',
    'unit_cost_eur', 'safety_stock',
    'otif_pct', 'risk_score_m5',
    'qty_deviation_pct', 'stock_coverage_weeks',
    'below_safety_stock', 'below_reorder',
    'supplier_risk_flag', 'delay_signal',
    # NEW features
    'total_confirmed_qty', 'n_customer_orders',
    'vip_order_count', 'at_risk_order_count',
    'customer_confirmation_rate', 'vip_order_ratio', 'at_risk_order_ratio',
    'avg_wc_utilization', 'avg_wc_overload', 'max_wc_utilization',
    'capacity_constraint_flag', 'high_downtime_flag'
]

TARGET_SHORTAGE = 'shortage_flag'

df_shortage = df[FEATURE_COLS_SHORTAGE + [TARGET_SHORTAGE]].dropna()
print(f'Shortage model dataset shape: {df_shortage.shape}')

X_s = df_shortage[FEATURE_COLS_SHORTAGE]
y_s = df_shortage[TARGET_SHORTAGE]

print(f'\nClass distribution:')
print(y_s.value_counts())

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_s, y_s, test_size=0.2, stratify=y_s, random_state=42
)

# Train base Random Forest
rf_base = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
print('\nTraining base RandomForestClassifier...')
rf_base.fit(X_train_s, y_train_s)

y_pred_s = rf_base.predict(X_test_s)
y_prob_s = rf_base.predict_proba(X_test_s)[:, 1]

print('\n📊 Random Forest (Uncalibrated) — Test Metrics')
print(classification_report(y_test_s, y_pred_s, target_names=['No Shortage', 'Shortage']))
print(f'ROC-AUC: {roc_auc_score(y_test_s, y_prob_s):.4f}')
print(f'Accuracy: {accuracy_score(y_test_s, y_pred_s):.4f}')

# Apply calibration
print('\n🔧 Applying Platt scaling for probability calibration...')
rf_calibrated = CalibratedClassifierCV(
    rf_base,
    method='sigmoid',
    cv=5
)
rf_calibrated.fit(X_train_s, y_train_s)

y_prob_cal = rf_calibrated.predict_proba(X_test_s)[:, 1]
y_pred_cal = (y_prob_cal > 0.5).astype(int)

print('\n📊 Random Forest (Calibrated) — Test Metrics')
print(classification_report(y_test_s, y_pred_cal, target_names=['No Shortage', 'Shortage']))
print(f'ROC-AUC: {roc_auc_score(y_test_s, y_prob_cal):.4f}')
print(f'Accuracy: {accuracy_score(y_test_s, y_pred_cal):.4f}')

# Feature importance
feat_imp = pd.Series(rf_base.feature_importances_, index=FEATURE_COLS_SHORTAGE).sort_values(ascending=False)
print(f'\nTop 10 features:')
print(feat_imp.head(10))

# Save calibrated model
with open(OUTPUT_DIR + 'model_shortage_unified.pkl', 'wb') as f:
    pickle.dump(rf_calibrated, f)
print('\n✅ model_shortage_unified.pkl saved (calibrated Random Forest)')

# ============= 5. SAVE CONFIGURATION AND METRICS =============
print('\n' + '=' * 80)
print('STEP 5: Saving configuration and metrics')
print('=' * 80)

# Save feature columns
feature_columns = {
    'regression': FEATURE_COLS_REG,
    'shortage': FEATURE_COLS_SHORTAGE
}
with open(OUTPUT_DIR + 'feature_columns.json', 'w') as f:
    json.dump(feature_columns, f, indent=2)

# Save training metrics
training_metrics = {
    'model_order_qty_forecast': {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'r2': round(r2, 4),
        'cv_r2_mean': round(cv_r2.mean(), 4),
        'cv_r2_std': round(cv_r2.std(), 4)
    },
    'model_shortage_unified': {
        'roc_auc': round(roc_auc_score(y_test_s, y_prob_cal), 4),
        'accuracy': round(accuracy_score(y_test_s, y_pred_cal), 4)
    }
}
with open(OUTPUT_DIR + 'training_metrics.json', 'w') as f:
    json.dump(training_metrics, f, indent=2)

print('✅ Feature columns saved to feature_columns.json')
print('✅ Training metrics saved to training_metrics.json')

print('\n' + '=' * 80)
print('TRAINING COMPLETE!')
print('=' * 80)
print('\nTraining Metrics Summary:')
print(json.dumps(training_metrics, indent=2))
print(f'\nAll artifacts saved to: {OUTPUT_DIR}')
print('\nArtifacts:')
print('  ✅ model_order_qty_forecast.pkl')
print('  ✅ model_shortage_unified.pkl')
print('  ✅ label_encoder_material.pkl')
print('  ✅ label_encoder_scenario.pkl')
print('  ✅ feature_columns.json')
print('  ✅ training_metrics.json')
