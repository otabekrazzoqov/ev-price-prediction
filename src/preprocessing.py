import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Preprocessing:
    def __int__(self, df):
        self.df = df.copy()

    def handling_missing_values(self):
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if self.df[col].dtype == "object":
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
        return self
    

    def encoding(self, target_col=None):
        encoder = LabelEncoder()
        cols = self.df.columns.tolist()
        for col in cols:
            if col == target_col:
                continue
            if col not in self.df.columns:
                continue
            if self.df[col].dtype == "object":
                if self.df[col].nunique() <= 5:
                    dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
                    self.df = pd.concat([self.df.drop(columns=col), dummies], axis=1)
                else:
                    self.df[col] = encoder.fit_transform(self.df[col])
        return self
    
    def scaling(self, target_col=None):
        scaler = StandardScaler()
        num_cols = self.df.select_dtypes(include=["int64","float64"]).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
        return self