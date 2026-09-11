import time
import uuid

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

log = structlog.get_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as exc:
            # Handlers that raise must still appear in the access log —
            # unhandled 5xx are exactly the requests operators need most.
            duration = round((time.perf_counter() - start) * 1000, 2)
            log.info(
                "request",
                method=request.method,
                path=request.url.path,
                status=500,
                duration_ms=duration,
                request_id=request_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        duration = round((time.perf_counter() - start) * 1000, 2)
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration,
            request_id=request_id,
        )
        response.headers["X-Request-ID"] = request_id
        return response
