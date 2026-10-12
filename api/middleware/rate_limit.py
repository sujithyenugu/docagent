"""Per-IP sliding-window rate limiter for DocumentAgent API.

Usage (as a FastAPI dependency inside a route handler):

    limiter = RateLimiter(local_rpm=10, cloud_rpm=1)

    @router.post("/extract")
    async def extract(request: Request, use_local: bool = Form(True), ...):
        limiter.check(request.client.host, is_local=use_local)
        ...

``check()`` raises ``HTTPException(429)`` when the limit is exceeded.
"""

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException


class RateLimiter:
    """Sliding-window rate limiter keyed by (client_ip, backend_type).

    Args:
        local_rpm:  Max requests per minute for the local backend (default 10).
        cloud_rpm:  Max requests per minute for the cloud backend (default 1).
        window_sec: Width of the sliding window in seconds (default 60).
    """

    def __init__(
        self,
        local_rpm: int = 10,
        cloud_rpm: int = 1,
        window_sec: int = 60,
    ) -> None:
        self._local_rpm = local_rpm
        self._cloud_rpm = cloud_rpm
        self._window_sec = window_sec
        self._windows: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, client_ip: str, *, is_local: bool) -> None:
        """Check the rate limit for *client_ip*.

        Raises:
            HTTPException: 429 Too Many Requests when the limit is exceeded.
        """
        limit = self._local_rpm if is_local else self._cloud_rpm
        backend = "local" if is_local else "cloud"
        key = f"{client_ip}:{backend}"
        now = time.monotonic()

        with self._lock:
            window = self._windows[key]
            cutoff = now - self._window_sec
            # Evict stale entries
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= limit:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Rate limit exceeded: {limit} requests/min "
                        f"for {backend} backend. Retry after 60 s."
                    ),
                )
            window.append(now)
