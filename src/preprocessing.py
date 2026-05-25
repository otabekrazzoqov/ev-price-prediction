from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

class Preprocessing:
    def __init__(self, df):                        
        self.df = df.copy()               

    def handling_missing_values(self):
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if self.df[col].dtype == "str":
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
        return self

    def encoding(self, target_col=None):
        cols = self.df.columns.tolist()
        for col in cols:
            if col == target_col:
                continue
            if self.df[col].dtype == "str":
                if self.df[col].nunique() <= 5:
                    dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int) 
                    self.df = pd.concat([self.df.drop(columns=col), dummies], axis=1)
                else:
                    le = LabelEncoder()                     
                    self.df[col] = le.fit_transform(self.df[col])
                    self.label_encoders[col] = le            
        return self

    def scaling(self, target_col=None):
        num_cols = self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        if target_col and target_col in num_cols:          
            num_cols.remove(target_col)

        self.scaler = StandardScaler()                        
        self.df[num_cols] = self.scaler.fit_transform(self.df[num_cols])
        return self

    def get_df(self):                                      
        return self.df.copy()
    