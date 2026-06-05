# notebooks/02_preprocessing.ipynb  OR  scripts/run_pipeline.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader
from preprocessing import Preprocessing
from sklearn.model_selection import train_test_split


# Loading data ______________________________________________________

loader = DataLoader(subfolder="raw")
loader.load("ev_market_2026.csv") \
      .validate() \
      .info()

df = loader.get_df()

# Splitting data before preprocessing _________________________________

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Fit preprocessor on train data only ___________________________________

preprocessor = Preprocessing(train_df)
preprocessor.handling_missing_values() \
             .encoding(target_col="price_usd") \
             .scaling(target_col="price_usd")
train_clean = preprocessor.get_df()

# Transform test data using the SAME fitted preprocessor _________________
# transform() reuses encoder/scaler fitted on train never refits. ________ 

test_clean = preprocessor.transform(test_df)



# Save both sets to data/final

train_loader = DataLoader(subfolder="preprocessed")
train_loader.df = train_clean
train_loader.save("train_clean.csv", subfolder="final")


test_loader = DataLoader()
test_loader.df = test_clean
test_loader.save("test_clean.csv", subfolder="final")


