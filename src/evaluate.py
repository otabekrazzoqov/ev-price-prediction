import os
import joblib
import numpy as np 
import pandas as pd 
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)


from logger import get_logger

logger = get_logger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")


class Evaluator:

    def __init__(self, model, X_test, y_test, feature_names: list = None):
        self.model = model
        self.X_test = X_test
        self.y_test = np.array(y_test)
        self.feature_names = feature_names or list(range(X_test.shape[1]))
        self.y_pred = model.predict(X_test)
        self._metrics = {}
        os.makedirs(REPORTS_DIR, exist_ok=True)
        logger.info(f"Evaluator initialized | test samples: {len(self.y_test)}")


    def compute_metrics(self) -> "Evaluator":
        """Compute MAE, RMSE, R², MAPE and log them."""
        mae  = mean_absolute_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        r2   = r2_score(self.y_test, self.y_pred)
        mape = mean_absolute_percentage_error(self.y_test, self.y_pred) * 100

        self._metrics = {
            "MAE":  round(mae,  2),
            "RMSE": round(rmse, 2),
            "R2":   round(r2,   4),
            "MAPE": round(mape, 2),
        }

        logger.info("Evaluation Results:")
        for k, v in self._metrics.items():
            logger.info(f"  {k:<6}: {v}")
        logger.info("─────────────────")
        return self


    def plot_residuals(self, save_path: str = None) -> "Evaluator":
        """Two-panel figure: left  — Actual vs Predicted scatter right — Residuals vs Predicted scatter """
        save_path = save_path or os.path.join(REPORTS_DIR, "residuals.png")
        residuals = self.y_test - self.y_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1 — Actual vs Predicted
        ax = axes[0]
        ax.scatter(self.y_test, self.y_pred, alpha=0.4, edgecolors="none", color="#4C72B0")
        lims = [min(self.y_test.min(), self.y_pred.min()),
                max(self.y_test.max(), self.y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual Price (USD)")
        ax.set_ylabel("Predicted Price (USD)")
        ax.set_title("Actual vs Predicted")
        ax.legend()

        # Panel 2 — Residuals
        ax2 = axes[1]
        ax2.scatter(self.y_pred, residuals, alpha=0.4, edgecolors="none", color="#DD8452")
        ax2.axhline(0, color="red", linewidth=1.5, linestyle="--")
        ax2.set_xlabel("Predicted Price (USD)")
        ax2.set_ylabel("Residual (Actual − Predicted)")
        ax2.set_title("Residuals vs Predicted")

        metrics_text = "\n".join(f"{k}: {v}" for k, v in self._metrics.items())
        fig.text(0.5, -0.02, metrics_text, ha="center", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Residual plot saved → {save_path}")
        return self


    def plot_feature_importance(self, top_n: int = 20,
                                save_path: str = None) -> "Evaluator":
        """
        Horizontal bar chart of feature importances.
        Works for tree-based models (feature_importances_) and
        linear models (coef_).  Skipped gracefully otherwise.
        """
        save_path = save_path or os.path.join(REPORTS_DIR, "feature_importance.png")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            title = "Feature Importances (tree-based)"
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_)
            title = "Feature Coefficients (|coef|)"
        else:
            logger.warning("Model has no feature_importances_ or coef_ — skipping plot.")
            return self

        indices = np.argsort(importances)[-top_n:]
        names   = [self.feature_names[i] for i in indices]
        vals    = importances[indices]

        fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
        ax.barh(names, vals, color="#4C72B0")
        ax.set_xlabel("Importance")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Feature importance plot saved → {save_path}")
        return self


    def get_metrics(self) -> dict:
        if not self._metrics:
            logger.warning("compute_metrics() has not been called yet.")
        return self._metrics

    def get_metrics_df(self) -> pd.DataFrame:
        return pd.DataFrame([self._metrics])

    # class-level helper 

    @classmethod
    def from_saved_model(cls, model_path: str, X_test, y_test,
                         feature_names: list = None) -> "Evaluator":
        """Load a .joblib model and return an Evaluator instance."""
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return cls(model, X_test, y_test, feature_names)

    


             