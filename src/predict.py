import os
import joblib
import pandas as pd
import numpy as np

from logger import get_logger

logger = get_logger(__name__)

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")


class Predictor:
    """
    Loads a saved model and runs predictions on new data.

    Two workflows are supported:

    A) Single-row prediction (e.g. from a web form / API call)
       ----------------------------------------------------------
       predictor = Predictor("best_model_random_forest.joblib")
       price = predictor.predict_one({
           "brand": "Tesla", "model": "Model 3", ...
       })

    B) Batch prediction (e.g. a whole CSV)
       ----------------------------------------------------------
       predictor = Predictor("best_model_random_forest.joblib")
       df_with_preds = predictor.predict_batch(df)

    NOTE: The input data must already be preprocessed + feature-engineered
          in the same way as the training data (use Preprocessing.transform()
          and FeatureEngineering before calling predict).
    """

    def __init__(self, model_filename: str = None, model_path: str = None):
        """
        Parameters
        ----------
        model_filename : str, optional
            Filename inside the models/ directory, e.g. "best_model_random_forest.joblib"
        model_path : str, optional
            Absolute path to the model file (overrides model_filename).
        """
        if model_path is None and model_filename is None:
            raise ValueError("Provide either model_filename or model_path.")

        path = model_path or os.path.join(MODELS_DIR, model_filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = joblib.load(path)
        self.model_path = path
        logger.info(f"Predictor loaded model from: {path}")

    # ── public API ──────────────────────────────────────────────────────────

    def predict_one(self, input_dict: dict) -> float:
        """
        Predict the price for a single vehicle described as a dict.

        Parameters
        ----------
        input_dict : dict
            Feature name → value, matching the training feature set.

        Returns
        -------
        float
            Predicted price in USD.
        """
        df = pd.DataFrame([input_dict])
        prediction = self.model.predict(df)[0]
        logger.info(f"Single prediction: ${prediction:,.2f}")
        return float(round(prediction, 2))

    def predict_batch(self, df: pd.DataFrame,
                      output_col: str = "predicted_price_usd") -> pd.DataFrame:
        """
        Predict prices for a DataFrame of vehicles.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix (preprocessed + feature-engineered, no target column).
        output_col : str
            Name for the new predictions column.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with an appended predictions column.
        """
        preds = self.model.predict(df)
        result = df.copy()
        result[output_col] = np.round(preds, 2)
        logger.info(
            f"Batch prediction complete | rows: {len(result)} | "
            f"mean predicted price: ${preds.mean():,.2f}"
        )
        return result

    def get_model(self):
        """Return the raw sklearn model object."""
        return self.model


# ── Convenience function ───────────────────────────────────────────────────────

def load_latest_model(models_dir: str = None):
    """
    Auto-load the most recently modified .joblib file from models/.
    Useful during development when you don't want to hardcode a filename.
    """
    models_dir = models_dir or MODELS_DIR
    files = [
        f for f in os.listdir(models_dir) if f.endswith(".joblib")
    ]
    if not files:
        raise FileNotFoundError(f"No .joblib models found in {models_dir}")

    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(models_dir, f)))
    logger.info(f"Auto-loading latest model: {latest}")
    return Predictor(model_filename=latest)
