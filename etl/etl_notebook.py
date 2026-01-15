####Load from Unity Catalog → Pandas
import pandas as pd

messages_pd = spark.table("bronze.device_messages_raw").toPandas()
tests_pd = spark.table("bronze.rapid_step_tests_raw").toPandas()
# ---- device_messages_raw cleanup ----
messages_pd.columns = messages_pd.columns.str.strip()

messages_pd["sensor_type"] = messages_pd["sensor_type"].astype(str).str.strip()
messages_pd["message_origin"] = messages_pd["message_origin"].astype(str).str.strip()

messages_pd["timestamp"] = pd.to_datetime(
    messages_pd["timestamp"], errors="coerce"
)

# ---- rapid_step_tests_raw cleanup ----
tests_pd.columns = tests_pd.columns.str.strip()

tests_pd["start_time"] = pd.to_datetime(
    tests_pd["start_time"], errors="coerce"
)

tests_pd["total_steps"] = (
    pd.to_numeric(tests_pd["total_steps"], errors="coerce")
    .fillna(0)
    .astype(int)
)
####Join on device_id
etl_df = tests_pd.merge(
    messages_pd,
    on="device_id",
    how="inner"
)
####Create a tidy table
tidy_df = etl_df[[
    "device_id",
    "start_time",
    "sensor_type",
    "message_origin",
    "total_steps"
]].copy()

tidy_df.rename(columns={
    "start_time": "test_start_time",
    "sensor_type": "item_name",
    "message_origin": "category",
    "total_steps": "quantity"
}, inplace=True)
####Top 5 “items” by quantity (sensor activity)
top_5_items = (
    tidy_df
    .groupby("item_name", as_index=False)["quantity"]
    .sum()
    .sort_values("quantity", ascending=False)
    .head(5)
)
####“Revenue by category” → Activity by message origin
activity_by_category = (
    tidy_df
    .groupby("category", as_index=False)["quantity"]
    .sum()
    .sort_values("quantity", ascending=False)
)
####Busiest hour of day (tests started)
tidy_df["hour"] = tidy_df["test_start_time"].dt.hour

busiest_hour = (
    tidy_df
    .groupby("hour", as_index=False)["quantity"]
    .sum()
    .sort_values("quantity", ascending=False)
    .head(1)
)
####Save results (MLflow — permission safe)
import mlflow
import tempfile
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def log_df(df, name):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        mlflow.log_artifact(f.name, artifact_path="etl_metrics")

log_df(top_5_items, f"top_5_items_{timestamp}.csv")
log_df(activity_by_category, f"activity_by_category_{timestamp}.csv")
log_df(busiest_hour, f"busiest_hour_{timestamp}.csv")

print("ETL metrics logged to MLflow artifacts")