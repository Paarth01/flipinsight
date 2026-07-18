"""Product listing, search, and filter endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from db.models import Product, get_db
from api.schemas import ProductListResponse, ProductOut

router = APIRouter(prefix="/products", tags=["products"])


@router.get("", response_model=ProductListResponse)
def list_products(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: str | None = None,
    brand: str | None = None,
    search: str | None = Query(None, description="Search in product name"),
    min_price: float | None = None,
    max_price: float | None = None,
    db: Session = Depends(get_db),
):
    q = db.query(Product)
    if category:
        q = q.filter(Product.category_l1 == category)
    if brand:
        q = q.filter(Product.brand == brand)
    if search:
        q = q.filter(Product.product_name.ilike(f"%{search}%"))
    if min_price is not None:
        q = q.filter(Product.retail_price >= min_price)
    if max_price is not None:
        q = q.filter(Product.retail_price <= max_price)

    total = q.count()
    items = q.offset((page - 1) * page_size).limit(page_size).all()

    return ProductListResponse(total=total, page=page, page_size=page_size, items=items)


@router.get("/{uniq_id}", response_model=ProductOut)
def get_product(uniq_id: str, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.uniq_id == uniq_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product
