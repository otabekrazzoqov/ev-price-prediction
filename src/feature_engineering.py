import pandas as pd
import numpy as np
from logger import get_logger

logger = get_logger(__name__)


class FeatureEngineering:
    """
    Creates new features from the existing columns.
    All operations are deterministic (no fitting needed),
    so the same method works for both train and test sets.

    Call order:
        fe = FeatureEngineering(df)
        fe.add_efficiency_features()
          .add_performance_score()
          .add_age_feature()
          .add_value_score()
        result = fe.get_df()
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        logger.info(f"FeatureEngineering initialized | shape: {self.df.shape}")

    def add_efficiency_features(self):
        """
        range_per_kwh : miles of range per kWh of battery.
        charge_rate   : charging speed relative to battery size (kW per kWh).
        """
        logger.info("Adding efficiency features...")

        self.df["range_per_kwh"] = (
            self.df["range_miles"] / self.df["battery_capacity_kwh"].replace(0, np.nan)
        ).round(4)

        self.df["charge_rate"] = (
            self.df["charging_speed_kw"] / self.df["battery_capacity_kwh"].replace(0, np.nan)
        ).round(4)

        logger.debug("Added: range_per_kwh, charge_rate")
        return self   # <-- required for method chaining

    def add_performance_score(self):
        """
        performance_score : composite score from horsepower, torque,
                            acceleration (lower=better), and top speed.
                            Each component min-max normalised to [0,1].
        """
        logger.info("Adding performance score...")

        hp  = self.df["horsepower"]
        tq  = self.df["torque_nm"]
        acc = self.df["acceleration_0_60_mph"]
        spd = self.df["top_speed_mph"]

        def minmax(s):
            rng = s.max() - s.min()
            return (s - s.min()) / rng if rng != 0 else pd.Series(0.5, index=s.index)

        self.df["performance_score"] = (
            minmax(hp)  * 0.30 +
            minmax(tq)  * 0.25 +
            (1 - minmax(acc)) * 0.25 +   # invert: lower acceleration time = faster
            minmax(spd) * 0.20
        ).round(4)

        logger.debug("Added: performance_score")
        return self

    def add_age_feature(self, current_year: int = 2026):
        """vehicle_age : years since the model year."""
        logger.info("Adding vehicle age feature...")
        self.df["vehicle_age"] = current_year - self.df["year"]
        logger.debug(f"Added: vehicle_age (base year {current_year})")
        return self

    def add_value_score(self):
        """value_score : range per dollar x1000 — higher = better value."""
        logger.info("Adding value score feature...")
        self.df["value_score"] = (
            self.df["range_miles"] / self.df["price_usd"].replace(0, np.nan) * 1000
        ).round(4)
        logger.debug("Added: value_score")
        return self

    def get_df(self) -> pd.DataFrame:
        logger.info(f"Returning feature-engineered dataframe | shape: {self.df.shape}")
        new_cols = ["range_per_kwh", "charge_rate", "performance_score",
                    "vehicle_age", "value_score"]
        present = [c for c in new_cols if c in self.df.columns]
        logger.info(f"New features added: {present}")
        return self.df.copy()
