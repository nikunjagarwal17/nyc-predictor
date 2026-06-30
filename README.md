# NYC Residential Sale Price Predictor

Predicts 2024 NYC residential sale prices using 2020–2023 data.
Evaluation metric: **RMSLE** (Root Mean Squared Log Error) — lower is better.

---

## Files

| File | What it does |
|------|-------------|
| `nyc_price_analysis.ipynb` | Full analysis + model training notebook |
| `predict.py` | Standalone script — runs everything and saves predictions |
| `run.sh` | Launcher script |
| `data/train_data.csv` | Training data (2020–2023 NYC sales, 112k rows) |
| `data/test_data.csv` | Test data (2024 sales, sale_price empty) |
| `submission/suggestion.csv` | Final predictions |

---

## How to Run

```bash
./run.sh install   # install Python dependencies
./run.sh notebook  # open Jupyter notebook in browser
./run.sh predict   # run predict.py directly (faster, no Jupyter)
```

---

## Notebook — Block by Block

### 1. Imports
Loads pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost, LightGBM.
XGBoost and LightGBM are wrapped in try/except so the notebook still runs if they're missing.

### 2. Load Data
Reads `train_data.csv` and `test_data.csv` with `parse_dates` so the sale date column is usable immediately.

### 3. EDA

**3.1 Basic Info** — `train.info()` and `train.describe()` to understand column types, null counts, and value ranges at a glance.

**3.2 Missing Values** — Bar chart showing which columns have nulls and what % is missing.
`gross_square_feet` and `land_square_feet` are missing ~32% each — handled with median imputation later.

**3.3 Sale Price Distribution** — Histogram of raw prices (right-skewed) vs log(price) (near-normal).
Since RMSLE = RMSE on log scale, we train models on log(price) and convert back at the end.

**3.4 Box Plot Analysis** — Side-by-side box plots per borough, raw vs log space.
Shows why we use **per-borough log-space IQR** fences instead of a global cutoff:
in raw space Manhattan's whiskers are enormous; in log space each borough looks balanced.

**3.5 IQR Fence Table** — Computes the actual lower/upper fences for each borough and prints
how many rows fall outside (about 6.2% overall).

**3.6 Before/After Chart** — Bar chart of % removed per borough, plus overlaid
distribution before and after removal.

**3.7 Price by Borough** — Median price bar chart (Manhattan ~3× outer boroughs) and sales
volume pie chart.

**3.8 Price Over Time** — Quarterly median price trend (2020–2023). Shows the COVID-era
price surge and 2022 cooling. Justifies adding `sale_year` and `sale_quarter` as features.

**3.9 Building Class** — Top building categories by volume and by median price.
Luxury co-ops and condos are priced 3–5× higher. Labels alone don't capture this ordering
— target encoding handles it better.

**3.10 Gross Sq Ft vs Price** — Scatter plot colored by borough.
Moderate correlation (~0.56) but noisy — log-transforming sqft helps linearize it.

**3.11 Year Built** — Distribution histogram and median price by decade built.
Buildings from 2000+ command a clear premium. Used to derive `building_age` feature.

**3.12 Inspection Score** — Distribution and median price per score bin.
No clean linear trend — the model will learn the interaction with borough on its own.

**3.13 Neighborhood Index** — Scatter vs log(price). Low linear correlation but tree
models pick up non-linear patterns.

**3.14 Geographic Heatmap** — Every property plotted at lat/lon, colored by log(price).
Manhattan is clearly the most expensive; prices fade outward. Justifies using
lat/lon + distance-to-Midtown as features.

**3.15 Correlation Heatmap** — Pearson correlations between all numeric columns.
Helps spot multicollinearity. `gross_square_feet` has the strongest linear link to price.

---

### 4. Feature Engineering
Single function `engineer_features()` applied identically to train and test:

| Feature | Why |
|---------|-----|
| `sale_year`, `sale_month`, `sale_quarter` | Capture market cycle |
| `building_age` | Newer buildings → premium |
| `log_gross_sqft`, `log_land_sqft` | Log-linearizes the size-price relationship |
| `has_gross_sqft`, `has_land_sqft` | Flag tells model which rows were imputed |
| `sqft_per_unit` | Density proxy |
| `dist_to_manhattan` | Continuous price gradient via Haversine formula |
| `borough_x_class_enc` | Manhattan condo ≠ Queens condo |
| `zip_code_num` | Some zip codes are dramatically more expensive |

---

### 5. Outlier Removal
Per-borough log-space IQR loop (inline, no abstraction).
Removes ~6.2% of training rows — arm's-length transfers, data entry errors,
and non-market sales.

---

### 6. Target Encoding
Replaces high-cardinality categoricals (`neighborhood`, `zip_code`,
`building_class_category`, `building_class_at_time_of_sale`) with their
**mean log(price)** computed from training data only.
Unseen test categories fall back to the global mean.
This is what the model actually uses to understand "how expensive is this neighborhood".

---

### 7. Feature Prep
- Assembles the final feature list (28 columns)
- Median imputation for remaining NaNs (year\_built, lat/lon)
- 85/15 train/validation split

---

### 8. Models

**LightGBM** — Fast gradient boosted trees, `num_leaves=255`, early stopping at 100 rounds.

**XGBoost** — Similar but different internal structure; errors less correlated with LightGBM,
so averaging them helps. Used 3000 max trees since 1000 wasn't enough to converge.

**Random Forest** — Third model, very different from boosted trees.
Adds variance reduction to the ensemble.

**Ensemble** — Simple `np.mean()` of all three models' log-space predictions.
Each model has different failure modes; averaging cancels some errors out.

---

### 9. Feature Importance
Bar chart of the 20 most important features from LightGBM (or XGBoost/RF if LightGBM
isn't available). Target-encoded columns typically dominate.

---

### 10. Diagnostics
- **Predicted vs Actual** scatter — should cluster along the diagonal
- **Residual histogram** — should be centered near 0 with low spread

---

### 11. Save Predictions
- Converts log predictions back to dollars with `expm1()`
- Saves `submission/suggestion.csv` with just `id` and `sale_price` columns
- Sanity check: overlaid distribution of training actuals vs test predictions
