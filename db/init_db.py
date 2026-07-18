"""Create tables and bulk-load the cleaned CSV into SQLite."""

import logging
from pathlib import Path

import pandas as pd

from db.models import Base, DB_PATH, Product, SessionLocal, engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "products_clean.csv"


def init_db(csv_path: Path = PROCESSED_PATH, reset: bool = True) -> None:
    if reset and DB_PATH.exists():
        logger.info("Removing existing DB at %s", DB_PATH)
        DB_PATH.unlink()

    Base.metadata.create_all(bind=engine)
    logger.info("Tables created")

    df = pd.read_csv(csv_path, parse_dates=["crawl_timestamp"])
    df = df.where(pd.notna(df), None)

    session = SessionLocal()
    try:
        records = df.to_dict(orient="records")
        session.bulk_insert_mappings(Product, records)
        session.commit()
        logger.info("Inserted %d products into %s", len(records), DB_PATH)
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
