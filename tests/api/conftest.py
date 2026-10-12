"""API test configuration.

Provides an autouse fixture that resets the in-memory rate limiter windows
between every test, so cloud-backend tests don't trip the 1 req/min limit
accumulated by earlier test runs.
"""

import pytest


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Clear all per-IP sliding windows on every rate limiter before each test."""
    from api.routers import extract, forms, grade

    extract._limiter._windows.clear()
    forms._limiter._windows.clear()
    grade._limiter._windows.clear()
    yield
    # Cleanup after test (belt-and-suspenders)
    extract._limiter._windows.clear()
    forms._limiter._windows.clear()
    grade._limiter._windows.clear()
