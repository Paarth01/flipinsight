"""FlipInsight API — entry point."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routers import analytics, predict, products, recommend

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database (SQLite or PostgreSQL)
    from db.init_db import init_db
    
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(1, max_retries + 1):
        try:
            logging.info("Initializing database (attempt %d/%d)...", attempt, max_retries)
            # Run with reset=False to avoid dropping existing Postgres data
            init_db(reset=False)
            logging.info("Database initialization successful!")
            break
        except Exception as e:
            logging.error("Database connection attempt %d failed: %s", attempt, e)
            if attempt == max_retries:
                logging.critical("Could not connect to database after %d attempts. Exiting.", max_retries)
                raise e
            time.sleep(retry_delay)
            
    yield
    # Shutdown: do nothing


app = FastAPI(
    title="FlipInsight API",
    description="Product analytics and ML inference API for the Flipkart e-commerce dataset",
    version="1.0.0",
    lifespan=lifespan,
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
