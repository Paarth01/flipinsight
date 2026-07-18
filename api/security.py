"""Security utilities for FlipInsight: API Key verification and Rate Limiting."""

import logging
import os
import time
from collections import defaultdict
from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# API Key security scheme
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# In-memory request log for rate limiting: (ip_address, route) -> list of timestamps
_request_history = defaultdict(list)


def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify the incoming X-API-Key header.
    Defaults to 'dev-secret-key' if FLIPINSIGHT_API_KEY environment variable is not defined.
    """
    expected_key = os.environ.get("FLIPINSIGHT_API_KEY", "dev-secret-key")
    if not api_key or api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail=f"Unauthorized: Invalid or missing {API_KEY_NAME} header."
        )
    return api_key


class RateLimiter:
    """
    FastAPI dependency for sliding-window rate limiting.
    Tracks requests by client IP address and route.
    """

    def __init__(self, requests: int, window_seconds: int):
        self.requests = requests
        self.window_seconds = window_seconds

    def __call__(self, request: Request):
        # Fallback to local address if client metadata is missing
        client_ip = request.client.host if request.client else "127.0.0.1"
        route_path = request.url.path
        key = (client_ip, route_path)

        now = time.time()
        # Filter out timestamps older than the sliding window
        cutoff = now - self.window_seconds
        _request_history[key] = [t for t in _request_history[key] if t > cutoff]

        # Check if rate limit exceeded
        if len(_request_history[key]) >= self.requests:
            logger.warning(
                "Rate limit exceeded for client %s on %s. Max allowed: %d per %ds",
                client_ip, route_path, self.requests, self.window_seconds
            )
            raise HTTPException(
                status_code=429,
                detail=f"Too Many Requests. Limit is {self.requests} requests per {self.window_seconds} seconds."
            )

        # Record this request
        _request_history[key].append(now)
