```
python 

# EV Price Prediction

## Problem
Predict electric vehicle prices based on technical specs.

## Dataset
- 2000 rows, 24 features
- Target: price_usd ($16K - $270K)

## Models Used
- Random Forest
- XGBoost
- LightGBM

## Results
| Model | R2 | MAE |
|---|---|---|
| RandomForest | 0.94 | 1823 |
| XGBoost | 0.96 | 1542 |

## How to run
pip install -r requirements.txt
python src/train.py
python src/predict.py
```
