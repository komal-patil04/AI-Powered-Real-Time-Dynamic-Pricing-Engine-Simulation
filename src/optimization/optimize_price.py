
import pandas as pd
import joblib


def calculate_recommended_price(current_price, predicted_demand, inventory_level, elasticity):

    price = current_price

    if predicted_demand > 150:
        price *= 1.15
    elif predicted_demand > 100:
        price *= 1.10
    elif predicted_demand < 50:
        price *= 0.90

    if inventory_level < 20:
        price *= 1.05

    if elasticity < -2:
        price *= 0.95

    return round(price, 2)


# Everything below only runs when this file is executed directly
# (python src/optimization/optimize_price.py) - NOT when something
# else just imports calculate_recommended_price from it. That's what
# lets consumer.py reuse the function above without re-reading the
# whole dataset / re-writing optimized_prices.csv every time it starts.
if __name__ == "__main__":

    # --------------------------------------------------
    # 1. READ FEATURE DATASET
    # --------------------------------------------------

    df = pd.read_csv(
        "data/final/feature_dataset.csv"
    )

    print("Feature dataset loaded successfully!")
    print("Rows:", len(df))

    # --------------------------------------------------
    # 2. LOAD MODEL
    # --------------------------------------------------

    model = joblib.load(
        "models/demand_model.pkl"
    )

    # --------------------------------------------------
    # 3. SELECT FEATURES
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 4. PREDICT DEMAND
    # --------------------------------------------------
    # No scaling needed - XGBoost's tree splits are invariant to a
    # per-feature linear rescaling like StandardScaler, so predicting
    # on the raw features gives the same result.

    df["predicted_demand"] = model.predict(X)

    # --------------------------------------------------
    # 5. RECOMMENDED PRICE
    # --------------------------------------------------

    df["recommended_price"] = df.apply(
        lambda row: calculate_recommended_price(
            row["current_price"],
            row["predicted_demand"],
            row["inventory_level"],
            row["avg_price_elasticity"],
        ),
        axis=1,
    )

    # --------------------------------------------------
    # 6. SELECT OUTPUT COLUMNS
    # --------------------------------------------------

    output_cols = [
        "product_id",
        "current_price",
        "predicted_demand",
        "recommended_price"
    ]

    result = df[output_cols]

    # --------------------------------------------------
    # 7. SAVE RESULT
    # --------------------------------------------------

    result.to_csv(
        "data/final/optimized_prices.csv",
        index=False
    )

    print("Price Optimization Completed!")

    print("\nSample optimized prices:")
    print(result.head())

    print(
        "\nOutput saved to "
        "data/final/optimized_prices.csv"
    )
