import glob
import time
time.sleep(20)

from kafka import KafkaConsumer
import pandas as pd
import joblib
import json
import os


TOPIC = "transactions"
LIVE_DATA_PATH = "data/live_data.csv"


# Load the trained model ONCE at startup - this is the real-time
# inference step. Each incoming transaction event gets scored live,
# it is NOT reading precomputed batch predictions.

model = joblib.load("models/demand_model.pkl")
scaler = joblib.load("models/scaler.pkl")


_part_files = glob.glob("data/final/feature_dataset.csv")

if not _part_files:
    raise FileNotFoundError(
        "No part files found in data/final/feature_dataset/ — "
        "has create_feature.py been run yet?"
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


def optimize_price(current_price, predicted_demand, inventory, elasticity):

    new_price = current_price

    if predicted_demand > 150:
        new_price *= 1.15
    elif predicted_demand > 100:
        new_price *= 1.10
    elif predicted_demand < 50:
        new_price *= 0.90

    if inventory < 20:
        new_price *= 1.05

    if elasticity < -2:
        new_price *= 0.95

    return round(new_price, 2)


def score_transaction(event: dict) -> dict:

    profit_margin = event["current_price"] - event["cost_price"]
    inventory_risk = int(event["inventory_level"] < 20)
    weekend = int(event["day_of_week"] >= 5)
    inventory_ratio = avg_inventory_ratio_by_product.get(
        event["product_id"], _feat_df["inventory_ratio"].mean()
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

    X_scaled = scaler.transform(row)
    predicted_demand = float(model.predict(X_scaled)[0])

    recommended_price = optimize_price(
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


def connect_consumer(retries=10, delay=5):

    for attempt in range(1, retries + 1):

        try:

            return KafkaConsumer(
                TOPIC,
                bootstrap_servers="kafka:29092",
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

if __name__ == "__main__":

    consumer = connect_consumer()
    #consumer = KafkaConsumer(
     #   TOPIC,
      #  bootstrap_servers='kafka:29092',
       # value_deserializer=lambda x: json.loads(x.decode())
    #)

    for message in consumer:

        event = message.value
        result = score_transaction(event)

        row_df = pd.DataFrame([result])
        write_header = not os.path.exists(LIVE_DATA_PATH)

        row_df.to_csv(
            LIVE_DATA_PATH,
            mode="a",
            header=write_header,
            index=False
        )

        print("Scored live:", result)