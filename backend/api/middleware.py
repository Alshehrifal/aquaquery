"""API middleware: CORS, rate limiting, request logging."""

import logging
import time
import uuid
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into each request."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log request method, path, and duration."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000
        request_id = getattr(request.state, "request_id", "unknown")
        logger.info(
            "request_id=%s method=%s path=%s status=%d duration_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter per client IP."""

    def __init__(self, app, requests_per_minute: int = 10) -> None:
        super().__init__(app)
        self._limit = requests_per_minute
        self._window = 60.0  # seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _clean_old_requests(self, ip: str, now: float) -> None:
        cutoff = now - self._window
        self._requests[ip] = [t for t in self._requests[ip] if t > cutoff]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for non-API routes
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        self._clean_old_requests(client_ip, now)

        remaining = self._limit - len(self._requests[client_ip])

        if remaining <= 0:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMITED",
                        "message": f"Rate limit exceeded. Maximum {self._limit} requests per minute.",
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(self._limit),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": "60",
                },
            )

        self._requests[client_ip].append(now)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining - 1)
        return response
