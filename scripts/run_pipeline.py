"""
  1. Load raw data
  2. Split into train / test  (before ANY processing — no leakage)
  3. Feature engineering
  4. Preprocessing  (fit on train, transform test)
  5. Train models + compare
  6. Evaluate best model  (plots saved to reports/)
  7. Save best model to models/
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import DataLoader
from feature_engineering import FeatureEngineering
from preprocessing import Preprocessing
from train import Trainer
from evaluate import Evaluator

from sklearn.model_selection import train_test_split


# ── 1. Load ───────────────────────────────────────────────────────────────────

loader = DataLoader(subfolder="raw")
loader.load("ev_market_2026.csv").validate().info()
df = loader.get_df()


# ── 2. Split BEFORE any processing ────────────────────────────────────────────

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"\nSplit → train: {train_df.shape} | test: {test_df.shape}")


# ── 3. Feature engineering (stateless — safe to apply to both sets) ───────────

fe_train = FeatureEngineering(train_df)
train_df = (fe_train
            .add_efficiency_features()
            .add_performance_score()
            .add_age_feature()
            .add_value_score()
            .get_df())

fe_test = FeatureEngineering(test_df)
test_df = (fe_test
           .add_efficiency_features()
           .add_performance_score()
           .add_age_feature()
           .add_value_score()
           .get_df())

print(f"After feature engineering → train: {train_df.shape} | test: {test_df.shape}")


# ── 4. Preprocessing ──────────────────────────────────────────────────────────
# Fit ONLY on train; transform test with the same fitted encoders/scaler.
# Drop high-cardinality ID-like columns that don't generalise.

DROP_COLS = ["brand", "model"]    # too many unique values to encode meaningfully
train_df = train_df.drop(columns=DROP_COLS, errors="ignore")
test_df  = test_df.drop(columns=DROP_COLS,  errors="ignore")

preprocessor = Preprocessing(train_df)
preprocessor.handling_missing_values() \
             .encoding(target_col="price_usd") \
             .scaling(target_col="price_usd")

train_clean = preprocessor.get_df()
test_clean  = preprocessor.transform(test_df)

print(f"After preprocessing → train: {train_clean.shape} | test: {test_clean.shape}")


# ── 5. Save processed data ────────────────────────────────────────────────────

train_loader = DataLoader()
train_loader.df = train_clean
train_loader.save("train_clean.csv", subfolder="final")

test_loader = DataLoader()
test_loader.df = test_clean
test_loader.save("test_clean.csv", subfolder="final")


# ── 6. Train ──────────────────────────────────────────────────────────────────

trainer = Trainer(train_clean, test_clean, target_col="price_usd")
trainer.train_all()

print("\n── Model Comparison ─────────────────────────────")
print(trainer.get_results().to_string())
print("─────────────────────────────────────────────────\n")


# ── 7. Evaluate best model ────────────────────────────────────────────────────

best_model = trainer.get_best_model()
X_test = test_clean.drop(columns=["price_usd"])
y_test = test_clean["price_usd"]

evaluator = Evaluator(best_model, X_test, y_test,
                      feature_names=X_test.columns.tolist())
evaluator.compute_metrics() \
         .plot_residuals() \
         .plot_feature_importance()

print("Metrics:", evaluator.get_metrics())


# ── 8. Save best model ────────────────────────────────────────────────────────

trainer.save_best_model()

print("\n✅ Pipeline complete.")
print("   Processed data  → data/final/")
print("   Best model      → models/")
print("   Evaluation plots→ reports/")
