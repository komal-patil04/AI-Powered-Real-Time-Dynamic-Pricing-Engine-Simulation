
from fastapi import FastAPI
import pandas as pd
from datetime import datetime

app = FastAPI()

@app.get("/health")
def health():
    return {
        "status":"running"
    }


@app.get("/live")
def live():

    df = pd.read_csv(
        "data/live_data.csv"
    )

    return df.tail(
        10
    ).to_dict(
        orient="records"
    )


@app.get("/products")
def products():

    df = pd.read_csv(
        "data/raw/product_catalog.csv"
    )

    return df[
        [
            "product_id",
            "product_name"
        ]
    ].to_dict(
        orient="records"
    )


@app.get(
    "/recommended-price/{product_id}"
)
def price(product_id):

    output_path = ( "data/final/optimized_prices.csv" )
    # Check if optimization output exists 
    try:
        df = pd.read_csv( output_path ) 
    except FileNotFoundError: 
        return { "error": "optimized_prices.csv not found — " "has the pricing pipeline run yet?" }

    

    matches = df[df["product_id"] == product_id]

    if matches.empty:
        return {"error": f"product_id {product_id} not found"}

    row = matches.iloc[0]

    return {

        "product":
        row["product_name"],

        "recommended_price":
        row[
            "recommended_price"
        ]
    }