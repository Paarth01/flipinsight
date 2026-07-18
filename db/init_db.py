"""Create tables and bulk-load the cleaned CSV into SQLite."""

import logging
from pathlib import Path

import pandas as pd

from db.models import Base, DB_PATH, Product, SessionLocal, engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "products_clean.csv"


def init_db(csv_path: Path = PROCESSED_PATH, reset: bool = True) -> None:
    from db.models import DATABASE_URL
    if reset and DATABASE_URL.startswith("sqlite") and DB_PATH.exists():
        logger.info("Removing existing SQLite DB at %s", DB_PATH)
        DB_PATH.unlink()

    Base.metadata.create_all(bind=engine)
    logger.info("Tables created / verified")

    session = SessionLocal()
    try:
        product_count = session.query(Product).count()
        if product_count > 0:
            logger.info("Database already populated with %d products. Skipping bulk insert.", product_count)
            return
    except Exception as e:
        logger.warning("Error checking product count: %s", e)

    logger.info("Populating database from CSV...")
    df = pd.read_csv(csv_path, parse_dates=["crawl_timestamp"])
    df = df.where(pd.notna(df), None)

    try:
        records = df.to_dict(orient="records")
        import math
        for r in records:
            for k, v in r.items():
                if isinstance(v, float) and math.isnan(v):
                    r[k] = None

        session.bulk_insert_mappings(Product, records)
        session.commit()
        logger.info("Inserted %d products into database", len(records))
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
