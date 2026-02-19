"""
Shared pytest configuration and fixtures for solaris tests.
"""

import warnings

import pytest


def is_tpu_available():
    """Check if TPU is available for JAX."""
    try:
        import jax

        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


# Create a marker for tests that require TPU
requires_tpu = pytest.mark.skipif(
    not is_tpu_available(), reason="TPU not available - skipping TPU-specific test"
)


@pytest.fixture(scope="session", autouse=True)
def warn_if_no_tpu():
    """Warn at the start of the test session if TPU is not available."""
    if not is_tpu_available():
        warnings.warn(
            "TPU is not available. TPU-specific tests will be skipped.", UserWarning
        )
    yield
