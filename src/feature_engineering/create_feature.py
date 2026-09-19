import pandas as pd
import glob
import os

# -----------------------------------
# 1. READ MASTER DATASET
# -----------------------------------

files = glob.glob(
    "data/processed/master_dataset/part-*.csv"
)

if not files:
    raise FileNotFoundError(
        "No master dataset found. "
        "Run preprocessing_spark.py first."
    )

df = pd.concat(
    [pd.read_csv(file) for file in files],
    ignore_index=True
)

# -----------------------------------
# 2. DATE FEATURES
# -----------------------------------

df["date"] = pd.to_datetime(df["date"])

df["day"] = df["date"].dt.day

df["month"] = df["date"].dt.month

# Monday = 0, Sunday = 6
df["day_of_week"] = df["date"].dt.dayofweek

df["weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# -----------------------------------
# 3. PRICE FEATURES
# -----------------------------------

df["discount"] = (
    df["base_price"] -
    df["current_price"]
)

df["discount_pct"] = (
    df["discount"] /
    df["base_price"]
) * 100

df["discount_pct"] = df["discount_pct"].fillna(0)

# -----------------------------------
# 4. INVENTORY FEATURES
# -----------------------------------

df["low_stock"] = (
    df["inventory_level"] < 20
).astype(int)

# Historical average units sold
# for each product

df = df.sort_values(
    ["product_id", "date"]
)

df["avg_past_units_sold"] = (
    df.groupby("product_id")["units_sold"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)

df["inventory_ratio"] = (
    df["avg_past_units_sold"] /
    (df["inventory_level"] + 1)
)

df["inventory_ratio"] = (
    df["inventory_ratio"]
    .fillna(0)
)

df = df.drop(
    columns=["avg_past_units_sold"]
)

# -----------------------------------
# 5. PROFIT FEATURES
# -----------------------------------

df["profit_per_unit"] = (
    df["current_price"] -
    df["cost_price"]
)

# -----------------------------------
# 6. ELASTICITY FEATURES
# -----------------------------------
df["avg_price_elasticity"] = ( df["avg_price_elasticity"].fillna(0) )

df["high_elasticity"] = (
    df["avg_price_elasticity"] < -1
).astype(int)

# -----------------------------------
# 7. DEMAND FEATURES
# -----------------------------------

df["high_sales"] = (
    df["units_sold"] > 100
).astype(int)

# -----------------------------------
# 8. SELECT MODEL FEATURES
# -----------------------------------

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

# Keep target + model features
feature_df = df[
    ["product_id", "units_sold"] + feature_cols
]

# -----------------------------------
# 9. SAVE FEATURE DATASET
# -----------------------------------

os.makedirs(
    "data/final",
    exist_ok=True
)

feature_df.to_csv(
    "data/final/feature_dataset.csv",
    index=False
)

print("Feature engineering completed!")

print(
    "Feature dataset shape:",
    feature_df.shape
)

print(
    feature_df.head()
)