"""
ETL pipeline for the Flipkart e-commerce sample dataset.

Raw data problems this module handles:
  - product_category_tree is a stringified single-element list, e.g.
        ["Clothing >> Women's Clothing >> Lingerie >> Shorts"]
  - product_specifications is a stringified Ruby-hash-ish dict, e.g.
        {"product_specification"=>[{"key"=>"Fabric", "value"=>"Cotton"}, ...]}
  - product_rating / overall_rating are strings, often "No rating available"
  - retail_price / discounted_price have nulls and occasional zeros
  - brand is often missing

Output: a clean, flat pandas DataFrame ready to load into the database.
"""

import ast
import re
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "flipkart_com-ecommerce_sample.csv"
PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "products_clean.csv"

MAX_CATEGORY_DEPTH = 4  # how many levels of the category tree to keep as columns


def _parse_category_tree(raw: str) -> list[str]:
    """Turn '["A >> B >> C"]' into ['A', 'B', 'C']."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = ast.literal_eval(raw)
        text = parsed[0] if isinstance(parsed, list) and parsed else str(parsed)
    except (ValueError, SyntaxError):
        text = raw
    text = text.strip().strip('"').strip("'")
    parts = [p.strip() for p in text.split(">>") if p.strip()]
    return parts


def _parse_specifications(raw: str) -> dict:
    """
    Turn Ruby-hash-style spec strings into a real Python dict of {key: value}.
    Entries without a 'key' are dropped (they're usually box-content descriptions).
    """
    if not isinstance(raw, str) or not raw.strip():
        return {}
    # Convert Ruby hash-rocket syntax to Python dict syntax
    py_like = raw.replace("=>", ":")
    try:
        parsed = ast.literal_eval(py_like)
    except (ValueError, SyntaxError):
        return {}
    specs = {}
    try:
        for item in parsed.get("product_specification", []):
            if isinstance(item, dict) and "key" in item and "value" in item:
                specs[item["key"].strip()] = str(item["value"]).strip()
    except AttributeError:
        return {}
    return specs


def _parse_rating(raw) -> float | None:
    """'No rating available' -> None, '4.5' -> 4.5"""
    if raw is None:
        return None
    raw = str(raw).strip()
    if not raw or "no rating" in raw.lower():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", *df.shape)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting cleaning pass")
    df = df.copy()

    # --- category tree -> level columns ---
    # A handful of crawled rows (~1.6%) have a malformed tree where no ">>"
    # separator was captured, so the "single category" is actually the full
    # product title (e.g. "Vishudh Printed Women's Straight Kurta"). Treating
    # these as real categories massively inflates category cardinality with
    # junk. We drop the category for depth==1 rows and treat it as missing.
    cat_lists = df["product_category_tree"].apply(_parse_category_tree)
    cat_lists = cat_lists.apply(lambda lst: [] if len(lst) == 1 else lst)
    for i in range(MAX_CATEGORY_DEPTH):
        df[f"category_l{i+1}"] = cat_lists.apply(lambda lst, i=i: lst[i] if len(lst) > i else None)
    df["category_depth"] = cat_lists.apply(len)

    # --- specifications -> dict, and pull out a few common useful fields ---
    spec_dicts = df["product_specifications"].apply(_parse_specifications)
    df["spec_count"] = spec_dicts.apply(len)

    # --- prices ---
    df["retail_price"] = pd.to_numeric(df["retail_price"], errors="coerce")
    df["discounted_price"] = pd.to_numeric(df["discounted_price"], errors="coerce")
    valid_price = (df["retail_price"] > 0) & (df["discounted_price"] > 0)
    df["discount_amount"] = (df["retail_price"] - df["discounted_price"]).where(valid_price)
    df["discount_pct"] = (100 * df["discount_amount"] / df["retail_price"]).where(valid_price)
    # guard against bad rows where discounted > retail (data entry errors)
    df.loc[df["discount_pct"] < 0, ["discount_amount", "discount_pct"]] = None

    # --- ratings ---
    df["product_rating_num"] = df["product_rating"].apply(_parse_rating)
    df["overall_rating_num"] = df["overall_rating"].apply(_parse_rating)
    df["has_rating"] = df["overall_rating_num"].notna()

    # --- brand cleanup ---
    df["brand"] = df["brand"].astype(str).str.strip()
    df.loc[df["brand"].isin(["nan", "", "None"]), "brand"] = None

    # --- description features ---
    df["description"] = df["description"].fillna("")
    df["description_length"] = df["description"].str.len()
    df["description_word_count"] = df["description"].str.split().str.len()

    # --- crawl timestamp ---
    df["crawl_timestamp"] = pd.to_datetime(df["crawl_timestamp"], errors="coerce", utc=True)

    # --- final column selection ---
    keep_cols = [
        "uniq_id", "pid", "product_name", "brand",
        "category_l1", "category_l2", "category_l3", "category_l4", "category_depth",
        "retail_price", "discounted_price", "discount_amount", "discount_pct",
        "product_rating_num", "overall_rating_num", "has_rating",
        "description", "description_length", "description_word_count",
        "spec_count", "is_FK_Advantage_product", "crawl_timestamp", "product_url",
    ]
    clean_df = df[keep_cols].copy()

    # drop rows with no usable price AND no category — junk rows
    before = len(clean_df)
    clean_df = clean_df.dropna(subset=["product_name"])
    clean_df = clean_df[clean_df["category_l1"].notna()]
    after = len(clean_df)
    logger.info("Dropped %d unusable rows (%d -> %d)", before - after, before, after)

    clean_df = clean_df.drop_duplicates(subset=["uniq_id"])
    logger.info("Final clean dataset: %d rows", len(clean_df))
    return clean_df


def run(raw_path: Path = RAW_PATH, out_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    df = load_raw(raw_path)
    clean_df = clean(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out_path, index=False)
    logger.info("Wrote cleaned data to %s", out_path)
    return clean_df


if __name__ == "__main__":
    run()
