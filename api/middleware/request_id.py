"""Middleware that injects X-Request-ID into every request/response pair.

If the incoming request already carries X-Request-ID the same value is
echoed back; otherwise a fresh UUID4 is generated.  The ID is stored on
``request.state.request_id`` so route handlers can embed it in responses.
"""

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
