"""SQLAlchemy ORM models for FlipInsight."""

import os
from pathlib import Path

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "flipinsight.db"

# Set DATABASE_URL to switch to Postgres, e.g.
#   postgresql://flipinsight:flipinsight@localhost:5432/flipinsight
# Defaults to local SQLite for zero-config runs.
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DB_PATH}")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uniq_id = Column(String, unique=True, index=True, nullable=False)
    pid = Column(String, index=True)
    product_name = Column(String, index=True)
    brand = Column(String, index=True, nullable=True)

    category_l1 = Column(String, index=True)
    category_l2 = Column(String, index=True, nullable=True)
    category_l3 = Column(String, nullable=True)
    category_l4 = Column(String, nullable=True)
    category_depth = Column(Integer)

    retail_price = Column(Float, nullable=True)
    discounted_price = Column(Float, nullable=True)
    discount_amount = Column(Float, nullable=True)
    discount_pct = Column(Float, nullable=True)

    product_rating_num = Column(Float, nullable=True)
    overall_rating_num = Column(Float, nullable=True)
    has_rating = Column(Boolean, default=False)

    description = Column(String, nullable=True)
    description_length = Column(Integer, nullable=True)
    description_word_count = Column(Integer, nullable=True)

    spec_count = Column(Integer, nullable=True)
    is_FK_Advantage_product = Column(Boolean, default=False)
    crawl_timestamp = Column(DateTime, nullable=True)
    product_url = Column(String, nullable=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
