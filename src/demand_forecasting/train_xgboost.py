
import pandas as pd
import joblib
import os

from xgboost import XGBRegressor
from sklearn.model_selection import \
train_test_split
from sklearn.preprocessing import \
StandardScaler
from sklearn.metrics import \
mean_absolute_error

# -----------------------------
# 1. READ FEATURE DATASET
# -----------------------------

df = pd.read_csv( "data/final/feature_dataset.csv" ) 
print("Feature dataset loaded successfully!") 
print("Rows:", len(df))


# -----------------------------
# 2. SELECT FEATURES
# -----------------------------

feature_cols = [
    "current_price",
    "inventory_level",
    "discount_pct",
    "profit_margin",
    "avg_price_elasticity",
    "inventory_risk",
    "weekend",
    "day_of_week",
    "month",
    "inventory_ratio",
    "profit_per_unit"
]

X = df[feature_cols]

y = df["units_sold"]

# -----------------------------
# 3. TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 4. SCALE FEATURES
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# -----------------------------
# 5. TRAIN XGBOOST
# -----------------------------

model = XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# -----------------------------
# 6. EVALUATE MODEL
# -----------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

print("MAE:", mae)


# -------------------------------------------------- 
#  7. CREATE MODELS FOLDER 
# -------------------------------------------------- 
os.makedirs( "models", exist_ok=True ) 

# -------------------------------------------------- 
# # 8. SAVE MODEL 
# # -------------------------------------------------- 
joblib.dump( model, "models/demand_model.pkl" ) 
print("Model training completed!") 
print("Model saved to models/demand_model.pkl") 