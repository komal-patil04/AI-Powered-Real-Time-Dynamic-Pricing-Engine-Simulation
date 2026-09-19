import os

os.environ["HADOOP_HOME"] = r"C:\Users\Admin\Desktop\Pricing_Engine\hadoop-3.3.6"
os.environ["hadoop.home.dir"] = r"C:\Users\Admin\Desktop\Pricing_Engine\hadoop-3.3.6"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Create Spark session
spark = (SparkSession.builder \
    .appName("PricingEnginePreprocessing") \
    .config("spark.hadoop.native.lib", "false")
    .getOrCreate())

# -----------------------------
# 1. READ DATA
# -----------------------------

sales = spark.read.csv(
    "data/raw/sales_transactions.csv",
    header=True,
    inferSchema=True
)

catalog = spark.read.csv(
    "data/raw/product_catalog.csv",
    header=True,
    inferSchema=True
)

elasticity = spark.read.csv(
    "data/raw/price_elasticity.csv",
    header=True,
    inferSchema=True
)

# -----------------------------
# 2. CLEAN SALES DATA
# -----------------------------

sales = sales.dropDuplicates()

sales = sales.withColumn(
    "date",
    F.to_date("date")
)

sales = sales.fillna({
    "base_price": 0,
    "current_price": 0,
    "inventory_level": 0,
    "units_sold": 0,
    "revenue": 0,
    "profit": 0
})

# -----------------------------
# 3. CLEAN CATALOG DATA
# -----------------------------

catalog = catalog.dropDuplicates()

catalog = catalog.fillna({
    "department": "Unknown",
    "aisle": "Unknown"
})

# -----------------------------
# 4. CLEAN ELASTICITY
# -----------------------------

elasticity = elasticity.fillna({
    "avg_price_elasticity": 0
})

# -----------------------------
# 5. JOIN SALES + CATALOG
# -----------------------------

catalog_selected = catalog.select(
    "product_id",
    "product_name",
    "department",
    "store_id"
)

master_df = sales.join(
    catalog_selected,
    on=["product_id", "product_name", "department", "store_id"],
    how="left"
)
# -----------------------------
# 6. JOIN ELASTICITY
# -----------------------------

master_df = master_df.join(
    elasticity.select(
        "product_id",
        "avg_price_elasticity"
    ),
    on="product_id",
    how="left"
)

# -----------------------------
# 7. CREATE BASIC FEATURES
# -----------------------------

master_df = master_df.withColumn(
    "profit_margin",
    F.col("current_price") - F.col("cost_price")
)

master_df = master_df.withColumn(
    "inventory_risk",
    (F.col("inventory_level") < 20).cast("int")
)

master_df = master_df.withColumn(
    "high_demand",
    (F.col("units_sold") > 100).cast("int")
)

# -----------------------------
# 8. SAVE MASTER DATASET
# -----------------------------

master_df.write \
.mode("overwrite") \
.option("header", True) \
.csv("data/processed/master_dataset")

print("Preprocessing completed!")
print("Rows:", master_df.count())

master_df.show(5)

spark.stop()