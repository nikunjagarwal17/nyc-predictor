"""
NYC Residential Sale Price Predictor
Standalone script — runs without Jupyter, outputs submission/suggestion.csv and submission.zip

Usage:
    python predict.py
"""

import os
import shutil
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = "data/"
SUBMIT_DIR = "submission/"
RANDOM_SEED = 42

FEATURE_COLS = [
    "borough", "latitude", "longitude",
    "residential_units", "commercial_units", "total_units",
    "log_gross_sqft", "log_land_sqft", "has_gross_sqft", "has_land_sqft",
    "year_built", "building_age",
    "inspection_score", "neighborhood_index",
    "sale_year", "sale_month", "sale_quarter",
    "building_class_category_enc", "building_class_at_time_of_sale_enc",
    "building_class_as_of_final_roll_enc",
    "tax_class_at_time_of_sale_enc", "tax_class_as_of_final_roll_enc",
    "neighborhood_enc",
]


# ── Feature Engineering ──────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["sale_year"]    = df["sale_date"].dt.year
    df["sale_month"]   = df["sale_date"].dt.month
    df["sale_quarter"] = df["sale_date"].dt.quarter

    df["building_age"] = (df["sale_year"] - df["year_built"]).clip(0, 300)

    df["log_gross_sqft"] = np.log1p(df["gross_square_feet"].fillna(0))
    df["log_land_sqft"]  = np.log1p(df["land_square_feet"].fillna(0))
    df["has_gross_sqft"] = df["gross_square_feet"].notna().astype(int)
    df["has_land_sqft"]  = df["land_square_feet"].notna().astype(int)

    cat_cols = [
        "building_class_category", "building_class_at_time_of_sale",
        "building_class_as_of_final_roll", "tax_class_at_time_of_sale",
        "tax_class_as_of_final_roll", "neighborhood",
    ]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])

    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    train = pd.read_csv(DATA_DIR + "train_data.csv", parse_dates=["sale_date"])
    test  = pd.read_csv(DATA_DIR + "test_data.csv",  parse_dates=["sale_date"])
    print(f"  Train: {train.shape}, Test: {test.shape}")

    print("Engineering features...")
    train_fe = engineer_features(train)
    test_fe  = engineer_features(test)

    # Remove invalid prices (zero / negative / transfer-only sales)
    train_clean = train_fe[train_fe["sale_price"] > 0].copy()
    print(f"  Valid training rows: {len(train_clean):,}")

    X      = train_clean[FEATURE_COLS]
    y      = np.log1p(train_clean["sale_price"])
    X_test = test_fe[FEATURE_COLS]

    imputer    = SimpleImputer(strategy="median")
    X_imp      = pd.DataFrame(imputer.fit_transform(X),      columns=FEATURE_COLS)
    X_test_imp = pd.DataFrame(imputer.transform(X_test),     columns=FEATURE_COLS)

    X_train, X_val, y_train, y_val = train_test_split(
        X_imp, y, test_size=0.15, random_state=RANDOM_SEED
    )

    # Try gradient boosting models, fall back to Random Forest
    model_used = None

    try:
        import lightgbm as lgb
        print("Training LightGBM...")
        model = lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.05, num_leaves=127,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED,
            n_jobs=-1, verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )
        model_used = "LightGBM"
    except ImportError:
        pass

    if model_used is None:
        try:
            import xgboost as xgb
            print("Training XGBoost...")
            model = xgb.XGBRegressor(
                n_estimators=1000, learning_rate=0.05, max_depth=7,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED,
                n_jobs=-1, early_stopping_rounds=50, eval_metric="rmse",
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
            model_used = "XGBoost"
        except ImportError:
            pass

    if model_used is None:
        print("Training Random Forest (fallback)...")
        model = RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)
        model_used = "RandomForest"

    val_preds = model.predict(X_val)
    rmsle = np.sqrt(np.mean((val_preds - y_val) ** 2))
    print(f"\n{model_used} validation RMSLE: {rmsle:.4f}")

    print("\nGenerating test predictions...")
    test_log_preds   = model.predict(X_test_imp)
    test_price_preds = np.clip(np.expm1(test_log_preds), 1, None)

    os.makedirs(SUBMIT_DIR, exist_ok=True)
    submission = pd.DataFrame({"id": test["id"], "sale_price": test_price_preds})
    submission.to_csv(SUBMIT_DIR + "suggestion.csv", index=False)
    print(f"Saved {SUBMIT_DIR}suggestion.csv  ({len(submission):,} rows)")

    shutil.copy("predict.py", SUBMIT_DIR + "predict.py")
    shutil.make_archive("submission", "zip", SUBMIT_DIR)
    print("Created submission.zip")
    print("\nDone! Upload submission.zip to score your predictions.")


if __name__ == "__main__":
    main()
