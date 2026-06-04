from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from logger import get_logger

logger = get_logger(__name__)

class Preprocessing:
    def __init__(self, df):                        
        self.df = df.copy()
        self.label_encoders = {}   
        logger.info(f"Preprocessing initialized | shape {self.df.shape}")             

    def handling_missing_values(self):
        logger.info("Handling missing values...")
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if self.df[col].dtype == "object":
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                    logger.debug(f" [{col}] filled with mode")
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                    logger.debug(f"[{col}] filled with mean")
        logger.info("Missing values handled successfully")
        return self

    def encoding(self, target_col=None):
        logger.info("Encoding categorical columns...")
        cols = self.df.columns.tolist()
        for col in cols:
            if col == target_col:
                continue
            if self.df[col].dtype == "object":
                if self.df[col].nunique() <= 5:
                    dummies = pd.get_dummies(self.df[col], prefix=col, dtype="int") 
                    self.df = pd.concat([self.df.drop(columns=col), dummies], axis=1)
                else:
                    le = LabelEncoder()                     
                    self.df[col] = le.fit_transform(self.df[col])
                    self.label_encoders[col] = le
        logger.info("Encoding complete")            
        return self

    def scaling(self, target_col=None):
        logger.info("Scaling numerical columns...")
        num_cols = self.df.select_dtypes(
            include=["int64", "float64","int16","int32", "int8"]
        ).columns.tolist()

        if target_col and target_col in num_cols:          
            num_cols.remove(target_col)

        if not num_cols:
            logger.warning("No numerical columns found to scale.")

        self.scaler = StandardScaler()                        
        self.df[num_cols] = self.scaler.fit_transform(self.df[num_cols])
        logger.info(f"Scaling complete | scaled {len(num_cols)} columns")
        return self

    def get_df(self):
        logger.info(f"Returning preprocessed dataframe | shape: {self.df.shape}")                                      
        return self.df.copy()
    