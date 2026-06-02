import argparse
import logging
from pyspark.sql import SparkSession, functions as F, types as T

# ------------------------------------------------------------------------------
# 🪵 Logger Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Coyote_Validator")

# ------------------------------------------------------------------------------
# 0️⃣ Parse Arguments
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Coyote Rate Confirmation Extractor Validator")
parser.add_argument(
    "--source_table",
    required=True,
    help="Spark table name to validate (e.g., logistics.silver.coyote)"
)
args = parser.parse_args()
source_table = args.source_table

# ------------------------------------------------------------------------------
# 1️⃣ Spark Session
# ------------------------------------------------------------------------------
spark = SparkSession.builder.appName("Coyote_Extractor_Validator").getOrCreate()

# ------------------------------------------------------------------------------
# 2️⃣ Schema Definition  (must match EXTRACTION_FIELDS in coyote_text_to_columns.py)
# ------------------------------------------------------------------------------
STOP_FIELDS_PREFIX = [
    "po", "customer", "address", "city", "state", "zipcode", "phone",
    "start_datetime", "end_datetime",
    "commodity", "weight_lbs", "pieces", "late_fee",
]

fields = [
    "source_file",
    # Broker
    "broker_name", "broker_phone", "broker_fax", "broker_email",
    "broker_address", "broker_city", "broker_state", "broker_zipcode",
    "broker_rep",
    # Load
    "loadConfirmationNumber", "equipment",
    # Financials
    "flatRate", "fuelSurcharge", "totalCarrierPay",
    # Carrier
    "carrier_name", "carrier_usdot", "carrier_phone", "carrier_fax", "carrier_email",
    # Pickups
    *[f"pickup_{p}_{i}" for p in STOP_FIELDS_PREFIX for i in range(1, 4)],
    # Deliveries
    *[f"delivery_{p}_{i}" for p in STOP_FIELDS_PREFIX for i in range(1, 4)],
    # Audit
    "processed_at",
]

schema = T.StructType([T.StructField(f, T.StringType()) for f in fields])

# ------------------------------------------------------------------------------
# 3️⃣ Ground Truth Record
#    Source: 1679676366011_CY_FL-GA.pdf  (Load 28861101)
# ------------------------------------------------------------------------------
truth_record = {
    # Broker
    "broker_name":    "Coyote Logistics, LLC",
    "broker_phone":   "877-626-9683",
    "broker_fax":     "+1 (847) 810 4891",
    "broker_email":   "Tamaz.Bazgadze@coyote.com",
    "broker_address": "960 Northpoint Parkway, Suite 150",
    "broker_city":    "Alpharetta",
    "broker_state":   "GA",
    "broker_zipcode": "30005",
    "broker_rep":     "Tamaz Bazgadze",
    # Load
    "loadConfirmationNumber": "28861101",
    "equipment":              "Van, 53'",
    # Financials
    "flatRate":        "276.60",
    "fuelSurcharge":   "323.40",
    "totalCarrierPay": "600.00",
    # Carrier
    "carrier_name":  "GTT Freight Corp",
    "carrier_usdot": "3723304",
    "carrier_phone": "",
    "carrier_fax":   "",
    "carrier_email": "gtt.expresscorp@gmail.com",
    # Pickup 1 — United Sugars, Clewiston FL
    "pickup_po_1":             "PO-1162791",
    "pickup_customer_1":       "United Sugars",
    "pickup_address_1":        "450 SONORA DRIVE GATE D",
    "pickup_city_1":           "Clewiston",
    "pickup_state_1":          "FL",
    "pickup_zipcode_1":        "33440",
    "pickup_phone_1":          "+1 (863) 902 2707",
    "pickup_start_datetime_1": "2023-03-29T08:00:00",
    "pickup_end_datetime_1":   "2023-03-29T13:00:00",
    "pickup_commodity_1":      "Miscellaneous",
    "pickup_weight_lbs_1":     "44455",
    "pickup_pieces_1":         "850",
    "pickup_late_fee_1":       "100",
    # Pickup 2 & 3 — none in this document
    "pickup_po_2": "", "pickup_customer_2": "", "pickup_address_2": "",
    "pickup_city_2": "", "pickup_state_2": "", "pickup_zipcode_2": "",
    "pickup_phone_2": "", "pickup_start_datetime_2": "", "pickup_end_datetime_2": "",
    "pickup_commodity_2": "", "pickup_weight_lbs_2": "", "pickup_pieces_2": "", "pickup_late_fee_2": "",
    "pickup_po_3": "", "pickup_customer_3": "", "pickup_address_3": "",
    "pickup_city_3": "", "pickup_state_3": "", "pickup_zipcode_3": "",
    "pickup_phone_3": "", "pickup_start_datetime_3": "", "pickup_end_datetime_3": "",
    "pickup_commodity_3": "", "pickup_weight_lbs_3": "", "pickup_pieces_3": "", "pickup_late_fee_3": "",
    # Delivery 1 — Batory Foods, Lithia Springs GA
    "delivery_po_1":             "PO-1162791",
    "delivery_customer_1":       "Batory Foods",
    "delivery_address_1":        "885 DOUGLAS HILLS RD",
    "delivery_city_1":           "Lithia Springs",
    "delivery_state_1":          "GA",
    "delivery_zipcode_1":        "30122",
    "delivery_phone_1":          "+1 (800) 282 3101 x5054",
    "delivery_start_datetime_1": "2023-03-30T09:30:00",
    "delivery_end_datetime_1":   "2023-03-30T09:30:00",
    "delivery_commodity_1":      "Miscellaneous",
    "delivery_weight_lbs_1":     "44455",
    "delivery_pieces_1":         "850",
    "delivery_late_fee_1":       "200",
    # Delivery 2 & 3 — none in this document
    "delivery_po_2": "", "delivery_customer_2": "", "delivery_address_2": "",
    "delivery_city_2": "", "delivery_state_2": "", "delivery_zipcode_2": "",
    "delivery_phone_2": "", "delivery_start_datetime_2": "", "delivery_end_datetime_2": "",
    "delivery_commodity_2": "", "delivery_weight_lbs_2": "", "delivery_pieces_2": "", "delivery_late_fee_2": "",
    "delivery_po_3": "", "delivery_customer_3": "", "delivery_address_3": "",
    "delivery_city_3": "", "delivery_state_3": "", "delivery_zipcode_3": "",
    "delivery_phone_3": "", "delivery_start_datetime_3": "", "delivery_end_datetime_3": "",
    "delivery_commodity_3": "", "delivery_weight_lbs_3": "", "delivery_pieces_3": "", "delivery_late_fee_3": "",
}

# source_file and processed_at are runtime-generated; skip from comparison
SKIP_FIELDS = {"source_file", "processed_at"}

# ------------------------------------------------------------------------------
# 4️⃣ Load Target Table
# ------------------------------------------------------------------------------
target_df = spark.table(source_table)

# ------------------------------------------------------------------------------
# 5️⃣ Normalize (trim + lowercase for fair comparison)
# ------------------------------------------------------------------------------
def normalize(df):
    string_cols = [c for c, t in df.dtypes if t == "string"]
    return df.select(*[
        F.trim(F.lower(F.col(c))).alias(c) if c in string_cols else F.col(c)
        for c in df.columns
    ])

target_df = normalize(target_df)

# ------------------------------------------------------------------------------
# 6️⃣ Find the record under test
# ------------------------------------------------------------------------------
load_id = truth_record["loadConfirmationNumber"]
target_rows = target_df.filter(
    F.col("loadConfirmationNumber") == load_id.lower()
).collect()

results = []

if not target_rows:
    logger.error(f"❌ No record found for loadConfirmationNumber={load_id}")
    for field in truth_record:
        if field in SKIP_FIELDS:
            continue
        results.append((field, "❌ Missing record", truth_record.get(field), None))
else:
    logger.info(f"✅ Record found for loadConfirmationNumber={load_id}")
    target_values = target_rows[0].asDict()

    for field, truth_val in truth_record.items():
        if field in SKIP_FIELDS:
            continue

        target_val = target_values.get(field, "")

        # Normalize both sides the same way
        norm_truth  = str(truth_val).strip().lower()  if truth_val  else ""
        norm_target = str(target_val).strip().lower() if target_val else ""

        status = "✅ Match" if norm_truth == norm_target else "❌ Mismatch"
        results.append((field, status, truth_val, target_val))

# ------------------------------------------------------------------------------
# 7️⃣ Log Results
# ------------------------------------------------------------------------------
logger.info(f"\n{'─'*80}")
logger.info(f"Validation report for Load {load_id}")
logger.info(f"{'─'*80}")
for field, status, truth, target in results:
    if status.startswith("✅"):
        logger.info(  f"{field:35} | {status:12} | expected='{truth}' | got='{target}'")
    else:
        logger.error( f"{field:35} | {status:12} | expected='{truth}' | got='{target}'")

# ------------------------------------------------------------------------------
# 8️⃣ Summary & Fail if errors
# ------------------------------------------------------------------------------
errors   = [r for r in results if r[1].startswith("❌")]
matches  = [r for r in results if r[1].startswith("✅")]

logger.info(f"\n{'─'*80}")
logger.info(f"Summary: {len(matches)} matched, {len(errors)} mismatched out of {len(results)} fields")

if errors:
    logger.error(f"Validation FAILED for {len(errors)} field(s):")
    for field, status, truth, target in errors:
        logger.error(f"  ✗ {field}: expected='{truth}'  →  got='{target}'")
    raise ValueError(f"Validation failed for {len(errors)} fields")
else:
    logger.info("✅ All fields match perfectly.")
