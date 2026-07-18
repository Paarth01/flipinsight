"""FlipInsight API — entry point."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from api.routers import analytics, predict, products, recommend

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(
    title="FlipInsight API",
    description="Product analytics and ML inference API for the Flipkart e-commerce dataset",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(products.router)
app.include_router(analytics.router)
app.include_router(predict.router)
app.include_router(recommend.router)

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "static"
app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(DASHBOARD_DIR / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}
