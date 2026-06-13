# EV Price Prediction

A supervised machine learning project that predicts the **market price of electric vehicles** based on technical specifications, performance metrics, and market data.

Built with a clean, modular pipeline covering data loading, feature engineering, preprocessing, training, evaluation, and interactive prediction.

---

## Results

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| **Gradient Boosting** ✅ | $2,743 | $3,937 | **0.9864** | 3.7% |
| Random Forest | $3,593 | $5,145 | 0.9768 | — |
| Ridge Regression | $7,402 | $11,156 | 0.8909 | — |

The best model (Gradient Boosting) predicts EV prices within **~3.7% error** on average across a price range of $16,000 – $270,000.

---

## Project Structure

```
ev-price-prediction/
│
├── data/
│   ├── raw/                    # Original dataset (ev_market_2026.csv)
│   ├── preprocessed/           # Intermediate outputs
│   └── final/                  # train_clean.csv, test_clean.csv
│
├── models/                     # Saved .joblib model files
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_pipeline.ipynb       # Full pipeline walkthrough
│   ├── 03_error_analysis.ipynb # Model error deep-dive with plots
│   └── 04_predict.ipynb        # Interactive price predictor
│
├── reports/                    # Auto-generated plots and charts
│   ├── residuals.png
│   ├── feature_importance.png
│   ├── error_by_category.png
│   ├── error_by_price_band.png
│   └── config_comparison.png
│
├── scripts/
│   └── run_pipeline.py         # End-to-end pipeline runner
│
├── src/
│   ├── config.py               # Paths and constants
│   ├── logger.py               # Centralised logging (file + console)
│   ├── data_loader.py          # DataLoader class
│   ├── feature_engineering.py  # FeatureEngineering class
│   ├── preprocessing.py        # Preprocessing class (fit + transform)
│   ├── train.py                # Trainer class
│   ├── evaluate.py             # Evaluator class
│   └── predict.py              # Predictor class
│
├── logs/                       # Daily log files (auto-created)
├── venv/                       # Virtual environment (not committed)
├── config.py                   # Project-level config
├── requirements.txt
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Source | `data/raw/ev_market_2026.csv` |
| Rows | 2,000 |
| Features | 24 |
| Target | `price_usd` |
| Price range | $16,394 – $269,775 |
| Brands | Audi, BMW, BYD, Fisker, Ford, GM/Chevrolet, Honda, Hyundai, Kia, Lucid, Mercedes, NIO, Polestar, Porsche, Rivian, Tesla, Toyota, Volkswagen, Volvo, Xiaomi |

### Feature Overview

| Feature | Type | Description |
|---|---|---|
| `year` | int | Model year (2020–2026) |
| `variant` | categorical | Base / Standard / Long Range / Performance / Premium |
| `battery_capacity_kwh` | float | Battery size in kWh |
| `range_miles` | float | EPA range in miles |
| `charging_speed_kw` | float | Max charging speed (kW) |
| `acceleration_0_60_mph` | float | 0–60 mph time (seconds) |
| `top_speed_mph` | float | Top speed (mph) |
| `horsepower` | float | Motor output (hp) |
| `torque_nm` | float | Torque in Newton-metres |
| `drive_type` | categorical | AWD / FWD / RWD |
| `body_type` | categorical | SUV / Sedan / Hatchback / Coupe / Truck / Van |
| `market_segment` | categorical | Budget / Mid-range / Premium / Luxury |
| `safety_rating` | int | NHTSA rating (3–5 stars) |
| `autopilot_level` | int | SAE autonomy level (0–3) |
| `country_of_origin` | categorical | US / Germany / China / South Korea / Japan / Sweden |
| `seating_capacity` | int | Number of seats |
| `cargo_volume_cubic_ft` | float | Cargo space (ft³) |
| `weight_kg` | float | Vehicle weight (kg) |
| `annual_sales_units` | int | Units sold per year |
| `customer_rating` | float | User rating (3.0–4.51) |
| `warranty_years` | int | Warranty duration (years) |

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/otabekrazzoqov/ev-price-prediction.git
cd ev-price-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
python scripts/run_pipeline.py
```

This will:
- Load and validate raw data
- Split into train / test sets
- Apply feature engineering
- Fit preprocessing (encoding + scaling) on train, transform test
- Train Ridge, Random Forest, and Gradient Boosting models
- Print a model comparison table
- Save the best model to `models/`
- Save evaluation plots to `reports/`

---

## Notebooks

Open notebooks in VS Code or Jupyter Lab:

```bash
jupyter lab
```

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Explore distributions, correlations, and outliers |
| `02_pipeline.ipynb` | Step-by-step walkthrough of the full pipeline |
| `03_error_analysis.ipynb` | Deep-dive into model errors with Seaborn, Matplotlib, and Plotly |
| `04_predict.ipynb` | Interactive form — input an EV's specs and get a price prediction |

### Importing `src/` classes in notebooks

Every notebook starts with:

```python
import sys, os
sys.path.append(os.path.join(os.getcwd(), "..", "src"))

from data_loader import DataLoader
from preprocessing import Preprocessing
# etc.
```

---

## Pipeline Architecture

```
raw CSV
   │
   ▼
DataLoader          — load, validate, save
   │
   ▼
train_test_split    — 80/20, random_state=42
   │
   ├──────────────────────┐
   ▼                      ▼
train_df              test_df
   │                      │
   ▼                      │
FeatureEngineering    FeatureEngineering
(fit + transform)     (transform only)
   │                      │
   ▼                      │
Preprocessing.fit()   Preprocessing.transform()
   │                      │
   ▼                      ▼
train_clean.csv       test_clean.csv  →  data/final/
   │
   ▼
Trainer               — Ridge, RandomForest, GradientBoosting
   │
   ▼
Evaluator             — MAE, RMSE, R², MAPE + plots → reports/
   │
   ▼
best_model.joblib     → models/
```

---

## Module Reference

### `DataLoader`
```python
from data_loader import DataLoader

loader = DataLoader(subfolder="raw")
loader.load("ev_market_2026.csv").validate().info()
df = loader.get_df()
loader.save("output.csv", subfolder="final")
```

### `FeatureEngineering`
```python
from feature_engineering import FeatureEngineering

fe = FeatureEngineering(df)
df = (fe.add_efficiency_features()   # range_per_kwh, charge_rate
        .add_performance_score()      # composite 0–1 score
        .add_age_feature()            # vehicle_age = 2026 - year
        .add_value_score()            # range_miles / price_usd × 1000
        .get_df())
```

### `Preprocessing`
```python
from preprocessing import Preprocessing

# Fit on train only
preprocessor = Preprocessing(train_df)
preprocessor.handling_missing_values() \
             .encoding(target_col="price_usd") \
             .scaling(target_col="price_usd")
train_clean = preprocessor.get_df()

# Transform test — never re-fits
test_clean = preprocessor.transform(test_df)
```

### `Trainer`
```python
from train import Trainer

trainer = Trainer(train_clean, test_clean, target_col="price_usd")
trainer.train_all()
print(trainer.get_results())     # comparison table
trainer.save_best_model()        # → models/best_model_*.joblib
model = trainer.get_best_model()
```

### `Evaluator`
```python
from evaluate import Evaluator

ev = Evaluator(model, X_test, y_test, feature_names=X_test.columns.tolist())
ev.compute_metrics() \
  .plot_residuals() \
  .plot_feature_importance()
print(ev.get_metrics())
```

### `Predictor`
```python
from predict import Predictor

predictor = Predictor("best_model_gradient_boosting.joblib")

# Single prediction
price = predictor.predict_one({"battery_capacity_kwh": 82, ...})

# Batch prediction
df_with_preds = predictor.predict_batch(df)
```

---

## Engineered Features

Five new features are created from the raw columns before preprocessing:

| Feature | Formula | Intuition |
|---|---|---|
| `range_per_kwh` | `range_miles / battery_capacity_kwh` | Efficiency: more range per kWh = better tech |
| `charge_rate` | `charging_speed_kw / battery_capacity_kwh` | How fast the battery fills relative to its size |
| `performance_score` | Weighted composite of HP, torque, acceleration, top speed | Single number for overall performance |
| `vehicle_age` | `2026 - year` | Newer cars command higher prices |
| `value_score` | `range_miles / price_usd × 1000` | Range per dollar — higher = better value |

---

## Key Design Decisions

**No data leakage** — `train_test_split` happens before any feature engineering or preprocessing. The `Preprocessing` class is fitted exclusively on training data and only `transform()`ed on test data, so no information from the test set influences the encoders or scaler.

**Unseen category handling** — `Preprocessing.transform()` gracefully handles categories in the test set that were not seen during training: missing one-hot columns are added as `0`, and unseen label-encoded values are replaced with the most frequent training class instead of crashing.

**Modular, chainable API** — Every class uses method chaining (`return self`) so pipeline steps read naturally as a sequence.

**Centralised logging** — All classes share the same `logger.py` setup. Logs are written to dated files in `logs/` and printed to the console simultaneously. Works in both terminal and Jupyter environments.

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
joblib
nbformat
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## Roadmap

- [ ] Deploy as a REST API with FastAPI
- [ ] Containerise with Docker
- [ ] Track experiments with MLflow
- [ ] Add cross-validation to `Trainer`
- [ ] Hyperparameter tuning with `GridSearchCV`
- [ ] Add a Streamlit web app for non-technical users

---

## Author

**Otabek Razzoqov** — ML Engineer  
GitHub: [github.com/otabekrazzoqov](https://github.com/otabekrazzoqov)

---

## License

This project is licensed under the MIT License.
