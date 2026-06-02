import os
import re
import logging
import argparse
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, trim, regexp_replace, when, to_timestamp, current_timestamp, input_file_name
from pyspark.sql.types import StringType

# ============================================================
# Logger
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Target schema: same business columns produced by text_to_columns.py
# ============================================================
EXTRACTION_FIELDS = [
    "source_file",
    "broker_name", "broker_phone", "broker_fax", "broker_address",
    "broker_city", "broker_state", "broker_zipcode", "broker_email",
    "loadConfirmationNumber", "totalCarrierPay",
    "carrier_name", "carrier_mc", "carrier_address", "carrier_city",
    "carrier_state", "carrier_zipcode", "carrier_phone",
    "carrier_fax", "carrier_contact",
    *[f"{p}_{i}" for p in [
        "pickup_customer", "pickup_address", "pickup_city",
        "pickup_state", "pickup_zipcode",
        "pickup_start_datetime", "pickup_end_datetime"
    ] for i in range(1, 4)],
    *[f"{p}_{i}" for p in [
        "delivery_customer", "delivery_address", "delivery_city",
        "delivery_state", "delivery_zipcode",
        "delivery_start_datetime", "delivery_end_datetime"
    ] for i in range(1, 4)],
    "processed_at"
]

DATETIME_COLUMNS = [
    *[f"pickup_start_datetime_{i}" for i in range(1, 4)],
    *[f"pickup_end_datetime_{i}" for i in range(1, 4)],
    *[f"delivery_start_datetime_{i}" for i in range(1, 4)],
    *[f"delivery_end_datetime_{i}" for i in range(1, 4)],
    "processed_at"
]

NUMERIC_COLUMNS = ["totalCarrierPay"]

# Optional aliases in case the CSV comes from another extractor/export with slightly different names.
COLUMN_ALIASES = {
    "load_number": "loadConfirmationNumber",
    "load": "loadConfirmationNumber",
    "load_id": "loadConfirmationNumber",
    "total_usd": "totalCarrierPay",
    "total": "totalCarrierPay",
    "carrier_pay": "totalCarrierPay",
    "broker": "broker_name",
    "carrier": "carrier_name",
}


def _safe_name(name: str) -> str:
    """Make column names Spark-friendly and predictable."""
    name = (name or "").strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9A-Za-z_]", "", name)
    return name


def normalize_column_names(df):
    """Normalize imported CSV headers and apply known aliases."""
    used = set()
    for old in df.columns:
        new = _safe_name(old)
        new = COLUMN_ALIASES.get(new, new)
        base = new
        n = 2
        while new in used:
            new = f"{base}_{n}"
            n += 1
        used.add(new)
        if old != new:
            df = df.withColumnRenamed(old, new)
    return df


def clean_text_columns(df):
    """Trim strings, convert NaN/null-like text to null, and collapse repeated whitespace."""
    null_tokens = ["", "nan", "none", "null", "nat", "n/a", "na"]
    for c in df.columns:
        df = df.withColumn(c, col(c).cast(StringType()))
        df = df.withColumn(c, trim(regexp_replace(col(c), r"[\r\n\t]+", " ")))
        df = df.withColumn(c, regexp_replace(col(c), r"\s+", " "))
        df = df.withColumn(c, when(col(c).isNull() | (trim(col(c)) == ""), None).otherwise(col(c)))
        df = df.withColumn(c, when(col(c).rlike("^(?i)(" + "|".join(null_tokens[1:]) + ")$"), None).otherwise(col(c)))
    return df


def add_missing_schema_columns(df):
    """Ensure the output has the same columns as the original extraction table."""
    for c in EXTRACTION_FIELDS:
        if c not in df.columns:
            df = df.withColumn(c, lit(None).cast(StringType()))
    return df


def cast_business_columns(df):
    """Cast dates and numeric fields after CSV ingestion."""
    for c in DATETIME_COLUMNS:
        if c in df.columns:
            # Handles ISO values like 2021-11-19T12:00:00 and common CSV timestamp variants.
            df = df.withColumn(
                c,
                when(col(c).isNull(), None)
                .otherwise(
                    when(to_timestamp(col(c), "yyyy-MM-dd'T'HH:mm:ss").isNotNull(), to_timestamp(col(c), "yyyy-MM-dd'T'HH:mm:ss"))
                    .when(to_timestamp(col(c), "yyyy-MM-dd HH:mm:ss").isNotNull(), to_timestamp(col(c), "yyyy-MM-dd HH:mm:ss"))
                    .when(to_timestamp(col(c), "MM/dd/yyyy HH:mm").isNotNull(), to_timestamp(col(c), "MM/dd/yyyy HH:mm"))
                    .when(to_timestamp(col(c), "MM/dd/yyyy hh:mm a").isNotNull(), to_timestamp(col(c), "MM/dd/yyyy hh:mm a"))
                    .otherwise(to_timestamp(col(c)))
                )
            )

    for c in NUMERIC_COLUMNS:
        if c in df.columns:
            df = df.withColumn(c, regexp_replace(col(c).cast(StringType()), r"[$,]", "").cast("double"))

    # Zip codes and MC should remain strings to preserve leading zeroes.
    for c in [x for x in df.columns if x.endswith("zipcode") or x == "carrier_mc"]:
        df = df.withColumn(c, col(c).cast(StringType()))

    return df


def transform_csv(df):
    """Main transformation pipeline for the new CSV format."""
    df = normalize_column_names(df)
    df = clean_text_columns(df)

    if "source_file" not in df.columns:
        df = df.withColumn("source_file", input_file_name())

    df = add_missing_schema_columns(df)

    # If the CSV does not have processed_at, stamp the execution time.
    df = df.withColumn("processed_at", when(col("processed_at").isNull(), current_timestamp()).otherwise(col("processed_at")))

    df = cast_business_columns(df)

    # Keep the original target columns first; preserve any extra CSV columns at the end for audit/debugging.
    extra_cols = [c for c in df.columns if c not in EXTRACTION_FIELDS]
    return df.select(*EXTRACTION_FIELDS, *extra_cols)


def read_csv(spark, source_path: str):
    """Read a CSV file or a folder of CSV files from DBFS/Volumes/local paths."""
    return (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "false")
        .option("multiLine", "true")
        .option("escape", '"')
        .option("quote", '"')
        .load(source_path)
    )


def main(p):
    spark = SparkSession.builder.appName("TruckR CSV Transformation").getOrCreate()
    logger.info("Starting TruckR CSV transformation process")
    logger.info(f"Reading CSV from: {p['source_path']}")

    df_raw = read_csv(spark, p["source_path"])
    logger.info(f"Rows detected: {df_raw.count()}")

    df_out = transform_csv(df_raw)

    logger.info(f"Writing transformed records to table: {p['target_table']}")
    (
        df_out.write.format("delta")
        .mode(p.get("write_mode", "overwrite"))
        .option("mergeSchema", "true")
        .saveAsTable(p["target_table"])
    )

    logger.info("TruckR CSV transformation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruckR CSV transformation parameters")
    parser.add_argument("--source_path", required=True, help="CSV file path or folder path, e.g. dbfs:/FileStore/exports/truckr_loads.csv")
    parser.add_argument("--target_table", required=True, help="Target Delta table, e.g. logistics.bronze.truckr_loads")
    parser.add_argument("--write_mode", default="overwrite", choices=["overwrite", "append"], help="Delta write mode")
    args = parser.parse_args()

    params = {
        "source_path": args.source_path,
        "target_table": args.target_table,
        "write_mode": args.write_mode,
    }
    main(params)
