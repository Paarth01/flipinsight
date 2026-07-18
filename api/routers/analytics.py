"""Analytics endpoints — aggregations for the dashboard."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from db.models import Product, get_db
from api.schemas import BrandStats, CategoryCount, PriceStats, SummaryStats

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary", response_model=SummaryStats)
def get_summary(db: Session = Depends(get_db)):
    total_products = db.query(func.count(Product.id)).scalar()
    total_brands = db.query(func.count(func.distinct(Product.brand))).filter(Product.brand.isnot(None)).scalar()
    total_categories = db.query(func.count(func.distinct(Product.category_l1))).scalar()
    avg_price = db.query(func.avg(Product.retail_price)).scalar() or 0
    avg_discount = db.query(func.avg(Product.discount_pct)).scalar() or 0
    with_rating = db.query(func.count(Product.id)).filter(Product.has_rating.is_(True)).scalar()

    return SummaryStats(
        total_products=total_products,
        total_brands=total_brands,
        total_categories=total_categories,
        avg_retail_price=round(avg_price, 2),
        avg_discount_pct=round(avg_discount, 2),
        pct_with_rating=round(100 * with_rating / total_products, 2) if total_products else 0,
    )


@router.get("/categories", response_model=list[CategoryCount])
def get_category_counts(limit: int = Query(15, ge=1, le=50), db: Session = Depends(get_db)):
    rows = (
        db.query(Product.category_l1, func.count(Product.id).label("count"))
        .filter(Product.category_l1.isnot(None))
        .group_by(Product.category_l1)
        .order_by(func.count(Product.id).desc())
        .limit(limit)
        .all()
    )
    return [CategoryCount(category=r[0], count=r[1]) for r in rows]


@router.get("/price-by-category", response_model=list[PriceStats])
def get_price_by_category(limit: int = Query(15, ge=1, le=50), db: Session = Depends(get_db)):
    rows = (
        db.query(
            Product.category_l1,
            func.avg(Product.retail_price).label("avg_price"),
            func.avg(Product.discount_pct).label("avg_discount"),
            func.count(Product.id).label("count"),
        )
        .filter(Product.category_l1.isnot(None), Product.retail_price.isnot(None))
        .group_by(Product.category_l1)
        .order_by(func.count(Product.id).desc())
        .limit(limit)
        .all()
    )
    return [
        PriceStats(
            category=r[0],
            avg_retail_price=round(r[1], 2) if r[1] else 0,
            avg_discount_pct=round(r[2], 2) if r[2] else 0,
            product_count=r[3],
        )
        for r in rows
    ]


@router.get("/top-brands", response_model=list[BrandStats])
def get_top_brands(limit: int = Query(15, ge=1, le=50), db: Session = Depends(get_db)):
    rows = (
        db.query(
            Product.brand,
            func.count(Product.id).label("count"),
            func.avg(Product.retail_price).label("avg_price"),
        )
        .filter(Product.brand.isnot(None))
        .group_by(Product.brand)
        .order_by(func.count(Product.id).desc())
        .limit(limit)
        .all()
    )
    return [BrandStats(brand=r[0], product_count=r[1], avg_price=round(r[2], 2) if r[2] else 0) for r in rows]


@router.get("/discount-distribution")
def get_discount_distribution(db: Session = Depends(get_db)):
    """Bucket products into discount ranges for a histogram."""
    buckets = [(0, 10), (10, 25), (25, 40), (40, 55), (55, 70), (70, 100)]
    result = []
    for low, high in buckets:
        count = (
            db.query(func.count(Product.id))
            .filter(Product.discount_pct >= low, Product.discount_pct < high)
            .scalar()
        )
        result.append({"range": f"{low}-{high}%", "count": count})
    return result
