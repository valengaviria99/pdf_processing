import os
import re
import logging
import argparse
from abc import ABC, abstractmethod
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import MapType, StringType

# ============================================================
# Logger
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Base Extractor
# ============================================================
class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> dict:
        pass


# ============================================================
# Coyote Rate Confirmation Extractor
# ============================================================
class CoyoteExtractor(BaseExtractor):
    """
    Extracts broker, carrier, pickup, and delivery data from
    Coyote Logistics Rate Confirmation documents converted to
    plain text via pdf_to_text_with_pymupdf4llm.py.

    Expected document structure (Markdown output from pymupdf4llm):
        - Header with 'Rate Confirmation' and 'Load XXXXXXXX'
        - 'Send invoices to:' block → broker info
        - 'Booked By' block → broker rep info
        - Agreement table → carrier name / USDOT / email
        - 'Stop N: Pick Up' blocks → pickup stops
        - 'Stop N: Delivery' blocks → delivery stops
        - Charges table → flat rate / fuel surcharge / total
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize(self, text: str) -> str:
        if not text:
            return ""
        # Collapse non-breaking spaces, tabs → single space
        text = re.sub(r"[ \t\u00A0]+", " ", text)
        # Normalise line endings
        text = re.sub(r"\r\n?", "\n", text)
        # Collapse 3+ blank lines → 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip trailing whitespace per line
        text = "\n".join(ln.rstrip() for ln in text.splitlines())
        return text.strip()

    def _parse_date(self, raw: str) -> str:
        """
        Accept several date/time formats and return ISO-8601 datetime string.
        Returns empty string on failure.
        """
        raw = raw.strip()
        # Formats seen in Coyote docs: 'Wed 03/29/2023', '03/29/2023', '03/22/2023 10:52'
        for fmt in (
            "%a %m/%d/%Y",
            "%m/%d/%Y",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y %I:%M %p",
        ):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%dT%H:%M:%S")
            except ValueError:
                continue
        return ""

    def _combine_dt(self, date_str: str, time_str: str) -> str:
        """Combine a bare date string with a bare time string."""
        date_str = date_str.strip()
        time_str = time_str.strip()
        if not date_str:
            return ""
        if not time_str:
            return self._parse_date(date_str)
        # Remove weekday prefix if present
        date_clean = re.sub(r"^[A-Za-z]{3}\s+", "", date_str)
        for fmt in (
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y %I:%M %p",
            "%m/%d/%Y %I:%M%p",
        ):
            try:
                return datetime.strptime(f"{date_clean} {time_str}", fmt).strftime("%Y-%m-%dT%H:%M:%S")
            except ValueError:
                continue
        return self._parse_date(date_str)

    # ------------------------------------------------------------------
    # Stop block splitter
    # ------------------------------------------------------------------
    def _split_stop_blocks(self, text: str, stop_type: str):
        """
        Return list of raw text blocks for each stop of *stop_type*
        ('Pick Up' | 'Delivery').
        """
        # Match 'Stop N: Pick Up' or 'Stop N: Delivery'
        pattern = re.compile(
            rf"Stop\s+\d+\s*:\s*{re.escape(stop_type)}",
            re.I,
        )
        markers = list(pattern.finditer(text))
        if not markers:
            return []

        # Boundary: next stop header OR end of text
        next_stop = re.compile(r"Stop\s+\d+\s*:", re.I)
        blocks = []
        for i, m in enumerate(markers):
            start = m.end()
            # Look for the next 'Stop N:' that comes after this one
            remaining = text[start:]
            following = next_stop.search(remaining)
            if following:
                blocks.append(remaining[: following.start()].strip())
            else:
                blocks.append(remaining.strip())
        return blocks

    # ------------------------------------------------------------------
    # Main extract
    # ------------------------------------------------------------------
    def extract(self, text: str) -> dict:
        data = {f: "" for f in EXTRACTION_FIELDS}
        if not text:
            return data

        text = self._normalize(text)

        # ── Load / Confirmation Number ─────────────────────────────────
        m = re.search(r"(?:Rate Confirmation\s+)?Load\s+(\d+)", text, re.I)
        if m:
            data["loadConfirmationNumber"] = m.group(1).strip()

        # ── Broker info (invoice address) ──────────────────────────────
        #   "Send invoices to:\n960 Northpoint Parkway\nSuite 150\nAlpharetta, GA 30005"
        inv = re.search(
            r"Send invoices? to[:\s]*(.*?)(?=\n\n|\Z)",
            text, re.I | re.S,
        )
        if inv:
            iblock = inv.group(1).strip()
            lines = [ln.strip() for ln in iblock.splitlines() if ln.strip()]

            # First non-empty line that is NOT an email is the company/address start
            # Try to get address, city/state/zip
            addr_m = re.search(
                r"(\d[\w\s#.,\-]+?)\s*\n\s*(?:Suite\s*\d+\s*\n\s*)?([A-Za-z\s]+),\s*([A-Z]{2})\s+(\d{5})",
                iblock, re.I,
            )
            if addr_m:
                data["broker_address"] = addr_m.group(1).strip()
                data["broker_city"]    = addr_m.group(2).strip()
                data["broker_state"]   = addr_m.group(3).strip()
                data["broker_zipcode"] = addr_m.group(4).strip()

            # Suite / secondary address line
            suite_m = re.search(r"(Suite\s*\d+)", iblock, re.I)
            if suite_m and data["broker_address"]:
                data["broker_address"] += f", {suite_m.group(1).strip()}"

        # Broker phone from the prominent header phone (877-6COYOTE)
        phone_m = re.search(r"(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})", text)
        if phone_m:
            data["broker_phone"] = phone_m.group(1).strip()

        # Broker name is always 'Coyote Logistics, LLC' in these docs
        if re.search(r"Coyote Logistics", text, re.I):
            data["broker_name"] = "Coyote Logistics, LLC"

        # Broker fax / email from 'Booked By' block
        booked = re.search(
            r"Booked By(.*?)(?=Load Requirements|Equipment Requirements|Notes|Stop\s+\d|$)",
            text, re.I | re.S,
        )
        if booked:
            bblock = booked.group(1)
            if em := re.search(r"[\w.\-+]+@[\w.\-]+\.\w+", bblock):
                data["broker_email"] = em.group(0).strip()
            if fax := re.search(r"Fax[:\s]*([\+\d\s\(\)\-]{7,})", bblock, re.I):
                data["broker_fax"] = fax.group(1).strip()
            if rep := re.search(r"^([A-Z][a-z]+\s+[A-Z][a-z]+)", bblock.strip(), re.M):
                data["broker_rep"] = rep.group(1).strip()

        # Also try the Agreement section for fax/email
        agree = re.search(r"Agreement(.*?)(?=\n\n[A-Z]|\Z)", text, re.I | re.S)
        if agree:
            ablock = agree.group(1)
            if not data["broker_fax"]:
                if fax := re.search(r"Fax\s+([\+\d\s\(\)\-]{7,})", ablock, re.I):
                    data["broker_fax"] = fax.group(1).strip()
            if not data["broker_email"]:
                if em := re.search(r"[\w.\-+]+@[\w.\-]+\.\w+", ablock):
                    data["broker_email"] = em.group(0).strip()

        # ── Carrier info (from Agreement table) ───────────────────────
        #   Carrier  GTT Freight Corp   |  Broker  Coyote Logistics, LLC
        #   USDOT    3723304            |  Rep     Tamaz Bazgadze
        #   Phone    None               |  ...
        #   Email    gtt.expresscorp@gmail.com
        #   Fax      None
        carrier_block = re.search(
            r"Carrier\s+([^\n]+)\n.*?USDOT\s+(\d+).*?(?:Phone\s+([^\n]+)\n)?.*?(?:Email\s+([\w.\-+@]+)\n)?.*?(?:Fax\s+([^\n]+))?",
            text, re.I | re.S,
        )
        if carrier_block:
            data["carrier_name"]  = carrier_block.group(1).strip()
            data["carrier_usdot"] = carrier_block.group(2).strip()
            raw_phone = (carrier_block.group(3) or "").strip()
            data["carrier_phone"] = "" if raw_phone.lower() == "none" else raw_phone
            data["carrier_email"] = (carrier_block.group(4) or "").strip()
            raw_fax = (carrier_block.group(5) or "").strip()
            data["carrier_fax"]   = "" if raw_fax.lower() == "none" else raw_fax

        # ── Total Carrier Pay ─────────────────────────────────────────
        total_m = re.search(r"Total\s+USD\s*\$?([\d,]+\.\d{2})", text, re.I)
        if total_m:
            data["totalCarrierPay"] = total_m.group(1).replace(",", "").strip()

        # ── Flat Rate & Fuel Surcharge ─────────────────────────────────
        flat_m = re.search(r"Flat Rate\s+[\d.]+\s+\$[\d.]+\s+\$([\d,]+\.\d{2})", text, re.I)
        if flat_m:
            data["flatRate"] = flat_m.group(1).replace(",", "").strip()

        fuel_m = re.search(r"Fuel Surcharge\s+[\d.]+\s+\$[\d.]+\s+\$([\d,]+\.\d{2})", text, re.I)
        if fuel_m:
            data["fuelSurcharge"] = fuel_m.group(1).replace(",", "").strip()

        # ── Equipment ─────────────────────────────────────────────────
        equip_m = re.search(r"Equipment\s+([\w,\s']+?)(?:\n|$)", text, re.I)
        if equip_m:
            data["equipment"] = equip_m.group(1).strip()

        # ── Pickups ───────────────────────────────────────────────────
        pickup_blocks = self._split_stop_blocks(text, "Pick Up")
        for idx, block in enumerate(pickup_blocks[:3], start=1):
            self._parse_stop(block, "pickup", idx, data)

        # ── Deliveries ────────────────────────────────────────────────
        delivery_blocks = self._split_stop_blocks(text, "Delivery")
        for idx, block in enumerate(delivery_blocks[:3], start=1):
            self._parse_stop(block, "delivery", idx, data)

        return data

    # ------------------------------------------------------------------
    def _parse_stop(self, block: str, stype: str, idx: int, data: dict):
        """
        Fill data dict for pickup/delivery stop *idx* from raw *block* text.

        Coyote block structure (after stop header removed):
            Pick Up / Delivery Numbers   PO-XXXXXXX
            Confirmation Numbers         None
            Facility  United Sugars
            Address   450 SONORA DRIVE\nGATE D\nClewiston, FL 33440
            Contact   None
            Phone     +1 (863) 902 2707
            Scheduled For / Appointment Scheduled For
            Wed 03/29/2023
            from 08:00 - 13:00   |  at 09:30
            Driver Work  No Touch
            ...
        """
        prefix = stype  # 'pickup' | 'delivery'

        # PO / reference numbers
        po_m = re.search(r"(?:Pick\s*Up|Delivery)\s+Numbers?\s+([\w\-]+)", block, re.I)
        if po_m:
            data[f"{prefix}_po_{idx}"] = po_m.group(1).strip()

        # Facility name
        fac_m = re.search(r"Facility\s+(.+)", block, re.I)
        if fac_m:
            data[f"{prefix}_customer_{idx}"] = fac_m.group(1).strip()

        # Address block: 'Address  <street>\n<optional extra line>\n<City, ST ZIP>'
        addr_m = re.search(
            r"Address\s+([\w\s#.,\-\/]+?)(?:,\s*([A-Za-z\s]+),\s*([A-Z]{2})\s+(\d{5})|\n\s*([A-Za-z\s]+),\s*([A-Z]{2})\s+(\d{5}))",
            block, re.I,
        )
        if addr_m:
            street = addr_m.group(1).strip().replace("\n", " ")
            # city/state/zip may be in group 2-4 or 5-7
            city  = (addr_m.group(2) or addr_m.group(5) or "").strip()
            state = (addr_m.group(3) or addr_m.group(6) or "").strip()
            zipcd = (addr_m.group(4) or addr_m.group(7) or "").strip()
            data[f"{prefix}_address_{idx}"] = street
            data[f"{prefix}_city_{idx}"]    = city
            data[f"{prefix}_state_{idx}"]   = state
            data[f"{prefix}_zipcode_{idx}"] = zipcd
        else:
            # Fallback: look for City, ST ZIP pattern anywhere in block
            csz = re.search(r"([A-Za-z\s]+),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)", block)
            if csz:
                data[f"{prefix}_city_{idx}"]    = csz.group(1).strip()
                data[f"{prefix}_state_{idx}"]   = csz.group(2).strip()
                data[f"{prefix}_zipcode_{idx}"] = csz.group(3).strip()

        # Phone
        ph_m = re.search(r"Phone\s+([\+\d\s\(\)\-x]{7,})", block, re.I)
        if ph_m:
            raw_ph = ph_m.group(1).strip()
            if raw_ph.lower() != "none":
                data[f"{prefix}_phone_{idx}"] = raw_ph

        # Date  –  'Scheduled For\nWed 03/29/2023\nfrom 08:00 - 13:00'
        #       or 'Appointment Scheduled For\nThu 03/30/2023\nat 09:30'
        date_m = re.search(
            r"(?:Appointment\s+)?Scheduled\s+For\s+([A-Za-z]{0,3}\s*\d{1,2}/\d{1,2}/\d{4})",
            block, re.I,
        )
        # Time range: 'from HH:MM - HH:MM'
        range_m = re.search(r"from\s+(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", block, re.I)
        # Single appointment time: 'at HH:MM'
        appt_m  = re.search(r"\bat\s+(\d{1,2}:\d{2})\b", block, re.I)

        if date_m:
            raw_date = date_m.group(1).strip()
            if range_m:
                data[f"{prefix}_start_datetime_{idx}"] = self._combine_dt(raw_date, range_m.group(1))
                data[f"{prefix}_end_datetime_{idx}"]   = self._combine_dt(raw_date, range_m.group(2))
            elif appt_m:
                appt_dt = self._combine_dt(raw_date, appt_m.group(1))
                data[f"{prefix}_start_datetime_{idx}"] = appt_dt
                data[f"{prefix}_end_datetime_{idx}"]   = appt_dt
            else:
                parsed = self._parse_date(raw_date)
                data[f"{prefix}_start_datetime_{idx}"] = parsed
                data[f"{prefix}_end_datetime_{idx}"]   = parsed

        # Commodity / weight / pieces
        comm_m = re.search(
            r"(?:Commodity\s+Exp\s*Wt\s+Pieces\s+)?(\w[\w\s]+?)\s+([\d,]+)\s+Lbs\s+(\d+)",
            block, re.I,
        )
        if comm_m:
            data[f"{prefix}_commodity_{idx}"]  = comm_m.group(1).strip()
            data[f"{prefix}_weight_lbs_{idx}"] = comm_m.group(2).replace(",", "").strip()
            data[f"{prefix}_pieces_{idx}"]     = comm_m.group(3).strip()

        # Late fee
        late_m = re.search(r"\$(\d+)\s+Late\s+Fee", block, re.I)
        if late_m:
            data[f"{prefix}_late_fee_{idx}"] = late_m.group(1).strip()


# ============================================================
# Schema fields
# ============================================================
STOP_FIELDS_PREFIX = [
    "po", "customer", "address", "city", "state", "zipcode", "phone",
    "start_datetime", "end_datetime",
    "commodity", "weight_lbs", "pieces", "late_fee",
]

EXTRACTION_FIELDS = [
    # Broker / load header
    "broker_name", "broker_phone", "broker_fax", "broker_email",
    "broker_address", "broker_city", "broker_state", "broker_zipcode",
    "broker_rep",
    "loadConfirmationNumber",
    "equipment",
    # Financials
    "flatRate", "fuelSurcharge", "totalCarrierPay",
    # Carrier
    "carrier_name", "carrier_usdot", "carrier_phone", "carrier_fax", "carrier_email",
    # Pickups (up to 3 stops)
    *[f"pickup_{p}_{i}" for p in STOP_FIELDS_PREFIX for i in range(1, 4)],
    # Deliveries (up to 3 stops)
    *[f"delivery_{p}_{i}" for p in STOP_FIELDS_PREFIX for i in range(1, 4)],
    # Audit
    "processed_at",
]


# ============================================================
# Spark UDF wrapper
# ============================================================
def extract_fields_udf():
    extractor = CoyoteExtractor()

    def _extract(text):
        result = extractor.extract(text or "")
        result["processed_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        return result

    return udf(_extract, MapType(StringType(), StringType()))


# ============================================================
# Main (parameterized)
# ============================================================
def main(p):
    spark = SparkSession.builder.appName("CoyoteExtraction").getOrCreate()
    logger.info("Starting Coyote Rate Confirmation extraction process")

    input_path = p["source_path"]
    df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.txt")
        .option("recursiveFileLookup", "false")
        .load(input_path)
        .select(col("_metadata.file_path").alias("source_file"), col("content"))
    )

    df = df.withColumn("text", col("content").cast("string")).drop("content")
    logger.info(f"Files detected: {df.count()}")

    extract_udf = extract_fields_udf()
    df = df.withColumn("extracted", extract_udf(col("text")))

    for field in EXTRACTION_FIELDS:
        df = df.withColumn(field, col("extracted").getItem(field))

    df = df.drop("text", "extracted")

    logger.info(f"Writing {df.count()} records to {p['target_table']}")
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("mergeSchema", "true")
        .saveAsTable(p["target_table"])
    )
    logger.info("Coyote extraction completed successfully.")


# ============================================================
# CLI entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coyote Rate Confirmation PDF Extraction")
    parser.add_argument("--source_path",  required=True, help="Folder with .txt files from pdf_to_text_with_pymupdf4llm.py")
    parser.add_argument("--target_table", required=True, help="Delta table to write results (e.g. logistics.silver.coyote)")
    args = parser.parse_args()
    main({"source_path": args.source_path, "target_table": args.target_table})
