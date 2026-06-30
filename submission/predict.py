"""
NYC Residential Sale Price Predictor — v2 (Improved)

Key improvements over v1:
- Target encoding (CV-safe) for neighborhood, zip_code, building_class_category
- zip_code added as a feature (was missing)
- Outlier removal: drop implausible prices > $10M and < $10k
- XGBoost/LightGBM n_estimators raised to 3000 with early stopping
- Ensemble: average log-predictions from all available models
- Price-per-sqft feature (where available)
- Borough × building_class interaction encoding

Usage:
    python predict.py
"""

import os
import shutil
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

DATA_DIR    = "data/"
SUBMIT_DIR  = "submission/"
RANDOM_SEED = 42

# ── Target Encoding (CV-safe to prevent leakage) ─────────────────────────────
def target_encode(train_col: pd.Series, train_target: pd.Series,
                  test_col: pd.Series, n_splits: int = 5,
                  smoothing: float = 20.0) -> tuple[pd.Series, pd.Series]:
    """
    Bayesian smoothed target encoding.
    Training encoding uses OOF predictions to prevent leakage.
    """
    global_mean = train_target.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    train_enc = np.zeros(len(train_col))
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train_col)):
        fold_map = (
            pd.Series(train_target.iloc[tr_idx].values, index=train_col.iloc[tr_idx])
            .groupby(level=0)
            .agg(['mean', 'count'])
        )
        # Bayesian smoothing: blend category mean with global mean
        fold_map['smoothed'] = (
            (fold_map['mean'] * fold_map['count'] + global_mean * smoothing)
            / (fold_map['count'] + smoothing)
        )
        train_enc[val_idx] = train_col.iloc[val_idx].map(fold_map['smoothed']).fillna(global_mean)

    # Test encoding: use all train data
    full_map = (
        pd.Series(train_target.values, index=train_col)
        .groupby(level=0)
        .agg(['mean', 'count'])
    )
    full_map['smoothed'] = (
        (full_map['mean'] * full_map['count'] + global_mean * smoothing)
        / (full_map['count'] + smoothing)
    )
    test_enc = test_col.map(full_map['smoothed']).fillna(global_mean)

    return pd.Series(train_enc, index=train_col.index), test_enc.reset_index(drop=True)


# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Date features
    df['sale_year']    = df['sale_date'].dt.year
    df['sale_month']   = df['sale_date'].dt.month
    df['sale_quarter'] = df['sale_date'].dt.quarter

    # Building age at time of sale
    df['building_age'] = (df['sale_year'] - df['year_built']).clip(0, 300)

    # Log-transformed size features
    df['log_gross_sqft'] = np.log1p(df['gross_square_feet'].fillna(0))
    df['log_land_sqft']  = np.log1p(df['land_square_feet'].fillna(0))

    # Missingness indicators (informative for tree models)
    df['has_gross_sqft'] = df['gross_square_feet'].notna().astype(int)
    df['has_land_sqft']  = df['land_square_feet'].notna().astype(int)

    # Price per sqft proxy: log_gross_sqft per unit
    df['sqft_per_unit'] = df['gross_square_feet'].fillna(0) / (df['total_units'].fillna(1) + 1)

    # Borough × building_class interaction (label encode combination)
    df['borough_x_class'] = (
        df['borough'].astype(str) + '_' + df['building_class_at_time_of_sale'].astype(str).str.strip()
    )
    le_bxc = LabelEncoder()
    df['borough_x_class_enc'] = le_bxc.fit_transform(df['borough_x_class'])

    # Zip code: numeric (NaN-safe)
    df['zip_code_num'] = pd.to_numeric(df['zip_code'], errors='coerce')

    # Label encode remaining low-complexity categoricals
    for col in ['tax_class_at_time_of_sale', 'tax_class_as_of_final_roll',
                'building_class_as_of_final_roll']:
        df[col + '_enc'] = LabelEncoder().fit_transform(df[col].astype(str).str.strip())

    return df


# ── Full Feature List (after target encoding is added externally) ─────────────
BASE_FEATURE_COLS = [
    'borough', 'latitude', 'longitude', 'zip_code_num',
    'residential_units', 'commercial_units', 'total_units',
    'log_gross_sqft', 'log_land_sqft', 'has_gross_sqft', 'has_land_sqft',
    'sqft_per_unit',
    'year_built', 'building_age',
    'inspection_score', 'neighborhood_index',
    'sale_year', 'sale_month', 'sale_quarter',
    'tax_class_at_time_of_sale_enc', 'tax_class_as_of_final_roll_enc',
    'building_class_as_of_final_roll_enc',
    'borough_x_class_enc',
    # target-encoded columns added below:
    'neighborhood_te', 'zip_code_te', 'building_class_category_te',
    'building_class_at_time_of_sale_te',
]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    train = pd.read_csv(DATA_DIR + "train_data.csv", parse_dates=["sale_date"])
    test  = pd.read_csv(DATA_DIR + "test_data.csv",  parse_dates=["sale_date"])
    print(f"  Train: {train.shape}, Test: {test.shape}")

    print("Engineering features...")
    train_fe = engineer_features(train)
    test_fe  = engineer_features(test)

    # ── Outlier removal ───────────────────────────────────────────────────────
    # Keep only plausible residential sales: $10k–$10M range
    # Arm's-length transfers, data errors often show up as $0 or extreme values
    before = len(train_fe)
    valid_mask = (train_fe['sale_price'] >= 10_000) & (train_fe['sale_price'] <= 10_000_000)
    train_clean = train_fe[valid_mask].copy()
    print(f"  Outlier removal: {before - len(train_clean):,} rows dropped "
          f"({len(train_clean):,} remain)")

    y = np.log1p(train_clean['sale_price'])

    # ── Target encoding (CV-safe on training, full-data on test) ─────────────
    print("Computing target encodings...")
    te_cols = {
        'neighborhood':                    'neighborhood_te',
        'zip_code':                        'zip_code_te',
        'building_class_category':         'building_class_category_te',
        'building_class_at_time_of_sale':  'building_class_at_time_of_sale_te',
    }
    for raw_col, enc_col in te_cols.items():
        tr_enc, te_enc = target_encode(
            train_clean[raw_col].astype(str),
            y,
            test_fe[raw_col].astype(str),
        )
        train_clean[enc_col] = tr_enc.values
        test_fe[enc_col]     = te_enc.values

    # ── Build X/y ─────────────────────────────────────────────────────────────
    X      = train_clean[BASE_FEATURE_COLS]
    X_test = test_fe[BASE_FEATURE_COLS]

    imputer    = SimpleImputer(strategy="median")
    X_imp      = pd.DataFrame(imputer.fit_transform(X),      columns=BASE_FEATURE_COLS)
    X_test_imp = pd.DataFrame(imputer.transform(X_test),     columns=BASE_FEATURE_COLS)

    X_train, X_val, y_train, y_val = train_test_split(
        X_imp, y, test_size=0.15, random_state=RANDOM_SEED
    )
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    # ── Train models ─────────────────────────────────────────────────────────
    val_preds_log  = []   # collect validation predictions for comparison
    test_preds_log = []   # collect test predictions for ensemble

    # 1. LightGBM
    try:
        import lightgbm as lgb
        print("\nTraining LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=255,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_samples=20,
            reg_alpha=0.05,
            reg_lambda=0.5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(200),
            ],
        )
        vp = lgb_model.predict(X_val)
        rmsle = np.sqrt(np.mean((vp - y_val) ** 2))
        print(f"  LightGBM val RMSLE: {rmsle:.4f}")
        val_preds_log.append(('LightGBM', vp, rmsle))
        test_preds_log.append(lgb_model.predict(X_test_imp))
    except ImportError:
        print("  LightGBM not available.")

    # 2. XGBoost
    try:
        import xgboost as xgb
        print("\nTraining XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.05,
            reg_lambda=0.5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            early_stopping_rounds=100,
            eval_metric='rmse',
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=200,
        )
        vp = xgb_model.predict(X_val)
        rmsle = np.sqrt(np.mean((vp - y_val) ** 2))
        print(f"  XGBoost val RMSLE: {rmsle:.4f}")
        val_preds_log.append(('XGBoost', vp, rmsle))
        test_preds_log.append(xgb_model.predict(X_test_imp))
    except ImportError:
        print("  XGBoost not available.")

    # 3. Random Forest (always available, used as diversifying ensemble member)
    print("\nTraining Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=400,
        max_depth=25,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    rf_model.fit(X_train, y_train)
    vp = rf_model.predict(X_val)
    rmsle = np.sqrt(np.mean((vp - y_val) ** 2))
    print(f"  Random Forest val RMSLE: {rmsle:.4f}")
    val_preds_log.append(('Random Forest', vp, rmsle))
    test_preds_log.append(rf_model.predict(X_test_imp))

    # ── Ensemble: weighted average in log space ────────────────────────────────
    # Weight each model by 1/RMSLE² so better models contribute more
    print("\n── Ensemble Results ────────────────────────────────────────")
    weights = np.array([1.0 / (r ** 2) for _, _, r in val_preds_log])
    weights /= weights.sum()

    for (name, _, r), w in zip(val_preds_log, weights):
        print(f"  {name:<15}  RMSLE={r:.4f}  weight={w:.3f}")

    ensemble_val  = sum(w * vp for (_, vp, _), w in zip(val_preds_log, weights))
    ensemble_test = sum(w * tp for tp, w in zip(test_preds_log, weights))

    ensemble_rmsle = np.sqrt(np.mean((ensemble_val - y_val) ** 2))
    print(f"\n  Ensemble val RMSLE : {ensemble_rmsle:.4f}")

    # ── Save submission ───────────────────────────────────────────────────────
    os.makedirs(SUBMIT_DIR, exist_ok=True)

    test_price_preds = np.clip(np.expm1(ensemble_test), 1, None)
    submission = pd.DataFrame({"id": test["id"], "sale_price": test_price_preds})
    submission.to_csv(SUBMIT_DIR + "suggestion.csv", index=False)
    print(f"\nSaved {SUBMIT_DIR}suggestion.csv  ({len(submission):,} rows)")

    shutil.copy("predict.py", SUBMIT_DIR + "predict.py")
    shutil.make_archive("submission", "zip", SUBMIT_DIR)
    print("Created submission.zip  — ready to upload!")


if __name__ == "__main__":
    main()
