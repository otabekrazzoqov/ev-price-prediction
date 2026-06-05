from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from logger import get_logger

logger = get_logger(__name__)


class Preprocessing:
    def __init__(self, df):
        self.df = df.copy()
        self.label_encoders = {}
        self.scaler = None
        self._dummy_columns = {}    # stores {col: [dummy_col_names]} for transform()
        self._target_col = None     # remembered so transform() can skip it
        logger.info(f"Preprocessing initialized | shape {self.df.shape}")

    # ------------------------------------------------------------------ #
    #  FIT methods  (call these on train data only)                        #
    # ------------------------------------------------------------------ #

    def handling_missing_values(self):
        logger.info("Handling missing values...")
        self._fill_values = {}  # store fill values for use in transform()

        for col in self.df.columns:
            if self.df[col].isnull().any():
                if pd.api.types.is_string_dtype(self.df[col]):
                #if self.df[col].dtype == "object":
                    fill_val = self.df[col].mode()[0]
                    self.df[col] = self.df[col].fillna(fill_val)
                    logger.debug(f"[{col}] filled with mode: {fill_val!r}")
                else:
                    fill_val = self.df[col].mean()
                    self.df[col] = self.df[col].fillna(fill_val)
                    logger.debug(f"[{col}] filled with mean: {fill_val:.4f}")
                self._fill_values[col] = fill_val  # remember for transform()

        logger.info("Missing values handled successfully")
        return self

    def encoding(self, target_col=None):
        logger.info("Encoding categorical columns...")
        self._target_col = target_col

        # Snapshot object columns BEFORE the loop to avoid iterating over
        # a DataFrame whose columns are changing mid-loop.
        object_cols = [
            col for col in self.df.columns
            if pd.api.types.is_string_dtype(self.df[col]) and col != target_col
        ]

        for col in object_cols:
            if self.df[col].nunique() <= 5:
                # FIX: dtype=int → produces 0/1 integers, not "object" strings
                dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
                self._dummy_columns[col] = dummies.columns.tolist()  # remember for transform()
                self.df = pd.concat([self.df.drop(columns=col), dummies], axis=1)
                logger.debug(f"[{col}] one-hot encoded → {self._dummy_columns[col]}")
            else:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
                logger.debug(f"[{col}] label encoded | {le.classes_.tolist()}")

        logger.info("Encoding complete")
        return self

    def scaling(self, target_col=None):
        logger.info("Scaling numerical columns...")
        if target_col:
            self._target_col = target_col

        # FIX: include int8/int32 so freshly created dummy (0/1) columns are scaled too
        num_cols = self.df.select_dtypes(
            include=["int8", "int32", "int64", "float32", "float64"]
        ).columns.tolist()

        if self._target_col and self._target_col in num_cols:
            num_cols.remove(self._target_col)

        if not num_cols:
            logger.warning("No numerical columns found to scale.")
            return self

        self._scaled_cols = num_cols  # remember column order for transform()
        self.scaler = StandardScaler()
        self.df[num_cols] = self.scaler.fit_transform(self.df[num_cols])
        logger.info(f"Scaling complete | {len(num_cols)} columns scaled")
        return self

    def get_df(self):
        logger.info(f"Returning preprocessed dataframe | shape: {self.df.shape}")
        return self.df.copy()

    # ------------------------------------------------------------------ #
    #  TRANSFORM method  (call this on test / validation data)             #
    # ------------------------------------------------------------------ #

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the encoders and scaler that were fitted on training data
        to a new DataFrame (e.g. test set).  Never re-fits anything.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame with the same columns as the training data.

        Returns
        -------
        pd.DataFrame
            Transformed copy — same structure as the fitted training output.
        """
        logger.info(f"Transforming new data | input shape: {df.shape}")
        out = df.copy()

        # 1. Fill missing values using train-derived fill values
        if hasattr(self, "_fill_values"):
            for col, fill_val in self._fill_values.items():
                if col in out.columns and out[col].isnull().any():
                    out[col] = out[col].fillna(fill_val)
                    logger.debug(f"[{col}] filled with train value: {fill_val!r}")

        # 2. Apply encoding
        for col, dummy_cols in self._dummy_columns.items():
            # One-hot encode with the same prefix
            dummies = pd.get_dummies(out[col], prefix=col, dtype=int)
            out = pd.concat([out.drop(columns=col), dummies], axis=1)

            # Ensure all dummy columns seen in training exist (add missing as 0)
            for dc in dummy_cols:
                if dc not in out.columns:
                    out[dc] = 0
                    logger.debug(f"[{dc}] missing dummy column added with 0")

            # Drop any extra categories not seen during training
            extra = [c for c in out.columns if c.startswith(col + "_") and c not in dummy_cols]
            if extra:
                out = out.drop(columns=extra)
                logger.debug(f"[{col}] dropped unseen categories: {extra}")

        for col, le in self.label_encoders.items():
            if col not in out.columns:
                logger.warning(f"[{col}] label-encoded column missing from input — skipping")
                continue
            # Handle unseen labels gracefully by mapping them to the most frequent class
            known = set(le.classes_)
            unseen = set(out[col].dropna().unique()) - known
            if unseen:
                fallback = le.classes_[0]
                logger.warning(f"[{col}] unseen labels {unseen} → replaced with '{fallback}'")
                out[col] = out[col].apply(lambda x: x if x in known else fallback)
            out[col] = le.transform(out[col])

        # 3. Apply scaling (same column order as training)
        if self.scaler is not None and hasattr(self, "_scaled_cols"):
            # Only scale columns that actually exist in the test set
            cols_to_scale = [c for c in self._scaled_cols if c in out.columns]
            missing_scale_cols = [c for c in self._scaled_cols if c not in out.columns]
            if missing_scale_cols:
                logger.warning(f"Columns expected for scaling not found: {missing_scale_cols}")
            out[cols_to_scale] = self.scaler.transform(out[cols_to_scale])
            logger.info(f"Scaling applied | {len(cols_to_scale)} columns scaled")

        logger.info(f"Transform complete | output shape: {out.shape}")
        return out

























































# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import pandas as pd
# from logger import get_logger

# logger = get_logger(__name__)

# class Preprocessing:
#     def __init__(self, df):                        
#         self.df = df.copy()
#         self.label_encoders = {}
#         self.scaler = None
#         self._dummy_columns = {}    # stores{col: [dummy_col_names]} for transform()
#         self._target_col = None     # remembered so transform() can skip it
#         logger.info(f"Preprocessing initialized | shape {self.df.shape}")             

#     def handling_missing_values(self):
#         logger.info("Handling missing values...")
#         self._fill_values = {}   # store fill values for use in transform()

#         for col in self.df.columns:
#             if self.df[col].isnull().any():
#                 if pd.api.types.is_string_dtype(self.df[col]):    #!!!!!!!!!!!!!!!!!
#                     self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
#                     logger.debug(f" [{col}] filled with mode")
#                 else:
#                     self.df[col] = self.df[col].fillna(self.df[col].mean())
#                     logger.debug(f"[{col}] filled with mean")
#                 self
#         logger.info("Missing values handled successfully")
#         return self

#     def encoding(self, target_col=None):
#         logger.info("Encoding categorical columns...")
#         cols = self.df.columns.tolist()
#         for col in cols:
#             if col == target_col:
#                 continue
#             #if self.df[col].dtype == "object":
#             if pd.api.types.is_string_dtype(self.df[col]) and col != target_col:  # !!!!
#                 if self.df[col].nunique() <= 5:
#                     dummies = pd.get_dummies(self.df[col], prefix=col, dtype="int") 
#                     self.df = pd.concat([self.df.drop(columns=col), dummies], axis=1)
#                 else:
#                     le = LabelEncoder()                     
#                     self.df[col] = le.fit_transform(self.df[col])
#                     self.label_encoders[col] = le
#         logger.info("Encoding complete")            
#         return self

#     def scaling(self, target_col=None):
#         logger.info("Scaling numerical columns...")
#         num_cols = self.df.select_dtypes(
#             include=["int64", "float64","int16","int32", "int8"]
#         ).columns.tolist()

#         if target_col and target_col in num_cols:          
#             num_cols.remove(target_col)

#         if not num_cols:
#             logger.warning("No numerical columns found to scale.")

#         self.scaler = StandardScaler()                        
#         self.df[num_cols] = self.scaler.fit_transform(self.df[num_cols])
#         logger.info(f"Scaling complete | scaled {len(num_cols)} columns")
#         return self

#     def get_df(self):
#         logger.info(f"Returning preprocessed dataframe | shape: {self.df.shape}")                                      
#         return self.df.copy()
    