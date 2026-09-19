import glob
import time
time.sleep(20)

from kafka import KafkaConsumer
import pandas as pd
import joblib
import json
import os
import sys


TOPIC = "transactions"
LIVE_DATA_PATH = "data/live_data.csv"


# --------------------------------------------------
# IMPORT PRICE OPTIMIZATION FUNCTION
# --------------------------------------------------

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "optimization")
)

from src.optimization.optimize_price import calculate_recommended_price


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load("models/demand_model.pkl")

# --------------------------------------------------
# LOAD FEATURE DATA
# --------------------------------------------------

_part_files = glob.glob("data/final/feature_dataset.csv")

if not _part_files:
    raise FileNotFoundError(
        "No feature dataset found. Has create_feature.py been run?"
    )

_feat_df = pd.concat(
    (pd.read_csv(f) for f in _part_files),
    ignore_index=True
)


avg_inventory_ratio_by_product = (
    _feat_df.groupby("product_id")["inventory_ratio"].mean().to_dict()
)


FEATURE_ORDER = [
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
    "profit_per_unit",
]


# --------------------------------------------------
# REAL-TIME SCORING
# --------------------------------------------------

def score_transaction(event: dict) -> dict:

    profit_margin = (
        event["current_price"] - event["cost_price"]
    )

    inventory_risk = int(
        event["inventory_level"] < 20
    )

    weekend = int(
        event["day_of_week"] >= 5
    )

    inventory_ratio = avg_inventory_ratio_by_product.get(
        event["product_id"],
        _feat_df["inventory_ratio"].mean()
    )

    profit_per_unit = profit_margin


    row = pd.DataFrame([{
        "current_price": event["current_price"],
        "inventory_level": event["inventory_level"],
        "discount_pct": event["discount_pct"],
        "profit_margin": profit_margin,
        "avg_price_elasticity": event["avg_price_elasticity"],
        "inventory_risk": inventory_risk,
        "weekend": weekend,
        "day_of_week": event["day_of_week"],
        "month": event["month"],
        "inventory_ratio": inventory_ratio,
        "profit_per_unit": profit_per_unit,
    }])[FEATURE_ORDER]


    # XGBoost prediction
    predicted_demand = float(
        model.predict(row)[0]
    )


    # Use pricing function from optimize_price.py
    recommended_price = calculate_recommended_price(
        event["current_price"],
        predicted_demand,
        event["inventory_level"],
        event["avg_price_elasticity"],
    )


    return {
        "product_id": event["product_id"],
        "product_name": event["product_name"],
        "current_price": event["current_price"],
        "predicted_demand": round(predicted_demand, 2),
        "recommended_price": recommended_price,
        "avg_price_elasticity": event["avg_price_elasticity"],
        "inventory_risk": inventory_risk,
        "timestamp": event["timestamp"],
    }


# --------------------------------------------------
# KAFKA CONSUMER
# --------------------------------------------------

def connect_consumer(retries=10, delay=5):

    for attempt in range(1, retries + 1):

        try:

            return KafkaConsumer(
                TOPIC,
                bootstrap_servers="kafka:29092",
                group_id="pricing-consumer",
                value_deserializer=lambda x: json.loads(x.decode())
            )

        except Exception as e:

            print(
                f"Kafka not ready (attempt {attempt}/{retries})"
            )

            print(e)

            time.sleep(delay)

    raise RuntimeError(
        "Could not connect to Kafka after retries."
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    consumer = connect_consumer()

    for message in consumer:

        event = message.value

        result = score_transaction(event)

        row_df = pd.DataFrame([result])

        write_header = not os.path.exists(
            LIVE_DATA_PATH
        )

        row_df.to_csv(
            LIVE_DATA_PATH,
            mode="a",
            header=write_header,
            index=False
        )

        print(
            "Scored live:",
            result
        )