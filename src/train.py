import os
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from logger import get_logger

logger = get_logger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")


# Helpers 

def _get_X_y(df: pd.DataFrame, target_col: str):
    """Split a DataFrame into features X and target y."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def _regression_metrics(y_true, y_pred) -> dict:
    """Return a dict of MAE, RMSE, R²."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


# Main Trainer class

class Trainer:
    """
    Trains one or more regression models on a preprocessed training set,
    evaluates them on a validation / test set, and saves the best model.

    Usage
    -----
        trainer = Trainer(train_df, test_df, target_col="price_usd")
        trainer.train_all()
        trainer.save_best_model()
        results = trainer.get_results()
    """

    # Models to try 
    MODELS = {
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=12,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=42
        ),
    }

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                 target_col: str = "price_usd"):
        self.target_col = target_col
        self.X_train, self.y_train = _get_X_y(train_df, target_col)
        self.X_test,  self.y_test  = _get_X_y(test_df,  target_col)
        self.trained_models = {}
        self.results = {}
        self.best_model_name = None
        logger.info(
            f"Trainer initialized | train: {self.X_train.shape} | "
            f"test: {self.X_test.shape} | target: {target_col}"
        )

    # ── public API 

    def train_all(self) -> "Trainer":
        """Train every model in MODELS and evaluate on the test set."""
        logger.info(f"Training {len(self.MODELS)} models...")
        for name, model in self.MODELS.items():
            self._train_one(name, model)

        # Pick the best model by R²
        self.best_model_name = max(self.results, key=lambda n: self.results[n]["R2"])
        logger.info(
            f"Best model: {self.best_model_name} | "
            f"R²={self.results[self.best_model_name]['R2']} | "
            f"RMSE={self.results[self.best_model_name]['RMSE']}"
        )
        return self

    def save_best_model(self, filename: str = None) -> "Trainer":
        """Persist the best model to models/ as a .joblib file."""
        if self.best_model_name is None:
            raise RuntimeError("Call train_all() before save_best_model().")

        os.makedirs(MODELS_DIR, exist_ok=True)
        filename = filename or f"best_model_{self.best_model_name}.joblib"
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump(self.trained_models[self.best_model_name], path)
        logger.info(f"Best model saved → {path}")
        return self

    def save_model(self, name: str, filename: str = None) -> "Trainer":
        """Persist any trained model by name."""
        if name not in self.trained_models:
            raise KeyError(f"Model '{name}' has not been trained yet.")
        os.makedirs(MODELS_DIR, exist_ok=True)
        filename = filename or f"{name}.joblib"
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump(self.trained_models[name], path)
        logger.info(f"Model '{name}' saved → {path}")
        return self

    def get_results(self) -> pd.DataFrame:
        """Return a DataFrame comparing all trained models."""
        df = pd.DataFrame(self.results).T
        df.index.name = "model"
        df = df.sort_values("R2", ascending=False)
        return df

    def get_best_model(self):
        """Return the best sklearn model object."""
        if self.best_model_name is None:
            raise RuntimeError("Call train_all() first.")
        return self.trained_models[self.best_model_name]

    # ── private

    def _train_one(self, name: str, model) -> None:
        logger.info(f"Training '{name}'...")
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        metrics = _regression_metrics(self.y_test, preds)
        self.trained_models[name] = model
        self.results[name] = metrics
        logger.info(
            f"[{name}] MAE={metrics['MAE']:,.2f} | "
            f"RMSE={metrics['RMSE']:,.2f} | R²={metrics['R2']}"
        )
