import time
time.sleep(20)

from kafka import KafkaProducer
import json
import random
import pandas as pd
from datetime import datetime

# Simulates a live POS feed: a new transaction event for a random
# product, using realistic per-product price/cost data from the
# catalog. This is RAW event data, not a precomputed prediction -
# the consumer does the actual ML inference in real time.

catalog = pd.read_csv("data/raw/product_catalog.csv")
elasticity = pd.read_csv("data/raw/price_elasticity.csv")[
    ["product_id", "avg_price_elasticity"]
]

products = catalog.merge(elasticity, on="product_id", how="left")
products["avg_price_elasticity"] = products["avg_price_elasticity"].fillna(-1.0)


#producer = KafkaProducer(

 #   bootstrap_servers='kafka:29092',

  #  value_serializer=lambda x:

   # json.dumps(x).encode()
#)

TOPIC = "transactions"
def connect_producer(retries=10, delay=5):

    for attempt in range(1, retries + 1):

        try:

            return KafkaProducer(
                bootstrap_servers="kafka:29092",
                value_serializer=lambda x: json.dumps(x).encode()
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
producer = connect_producer()

while True:

    p = products.sample(1).iloc[0]

    # Simulate a plausible live discount and inventory snapshot
    discount_pct = random.choice([0, 0, 0, 5, 10, 15])
    current_price = round(p["base_price"] * (1 - discount_pct / 100), 2)
    inventory_level = random.randint(5, 150)

    now = datetime.now()

    message = {
        "product_id": int(p["product_id"]),
        "product_name": p["product_name"],
        "cost_price": float(p["cost_price"]),
        "base_price": float(p["base_price"]),
        "current_price": current_price,
        "discount_pct": discount_pct,
        "inventory_level": inventory_level,
        "avg_price_elasticity": float(p["avg_price_elasticity"]),
        "day_of_week": now.weekday(),
        "month": now.month,
        "timestamp": now.isoformat(),
    }

    producer.send(
        TOPIC,
        message
    )

    print("Sent transaction:", message)

    time.sleep(5)