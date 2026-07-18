"""Pydantic request/response models for the API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ProductOut(BaseModel):
    id: int
    uniq_id: str
    product_name: str
    brand: str | None = None
    category_l1: str | None = None
    category_l2: str | None = None
    retail_price: float | None = None
    discounted_price: float | None = None
    discount_pct: float | None = None
    overall_rating_num: float | None = None
    product_url: str | None = None

    model_config = ConfigDict(from_attributes=True)


class ProductListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[ProductOut]


class CategoryCount(BaseModel):
    category: str
    count: int


class PriceStats(BaseModel):
    category: str
    avg_retail_price: float
    avg_discount_pct: float
    product_count: int


class BrandStats(BaseModel):
    brand: str
    product_count: int
    avg_price: float


class SummaryStats(BaseModel):
    total_products: int
    total_brands: int
    total_categories: int
    avg_retail_price: float
    avg_discount_pct: float
    pct_with_rating: float


class PricePredictionRequest(BaseModel):
    category_l1: str = Field(..., examples=["Clothing"])
    category_l2: str | None = Field(None, examples=["Women's Clothing"])
    brand: str | None = Field(None, examples=["Nike"])
    description: str = Field(..., examples=["Cotton casual t-shirt for men, regular fit, machine washable"])
    is_FK_Advantage_product: bool = False


class PricePredictionResponse(BaseModel):
    predicted_retail_price: float
    currency: str = "INR"


class CategoryPredictionRequest(BaseModel):
    description: str = Field(..., examples=["Men's running shoes with breathable mesh upper and cushioned sole"])


class CategoryPredictionResponse(BaseModel):
    predicted_category: str
    confidence: float
    top_3: list[dict]


class SimilarProduct(BaseModel):
    uniq_id: str
    product_name: str
    brand: str | None = None
    category_l1: str | None = None
    retail_price: float | None = None
    discounted_price: float | None = None
    product_url: str | None = None
    similarity: float


class RecommendBySearchRequest(BaseModel):
    query: str = Field(..., examples=["waterproof running shoes for men"])
    k: int = Field(10, ge=1, le=25)


class RecommendResponse(BaseModel):
    source: str = Field(..., description="Either the source uniq_id or the search query text")
    results: list[SimilarProduct]
