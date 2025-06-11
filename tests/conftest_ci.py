"""CI-specific pytest configuration to force mocking."""

import os
import pytest

# Force CI mode
os.environ["CI"] = "true"
os.environ["PYTEST_RUNNING"] = "1"


def pytest_configure(config):
    """Configure pytest for CI environment."""
    # Add CI-specific markers
    config.addinivalue_line("markers", "ci: mark test to run only in CI")

    # Set environment variables
    os.environ["FLAME_CONFIG"] = "ci"
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "mock-token-for-ci"


def pytest_collection_modifyitems(config, items):
    """Modify test collection for CI."""
    skip_ollama = pytest.mark.skip(reason="Ollama not available in CI")
    skip_api = pytest.mark.skip(reason="API access not available in CI")

    for item in items:
        # Skip tests that require Ollama
        if "requires_ollama" in item.keywords:
            item.add_marker(skip_ollama)

        # Skip tests that require real API access
        if "requires_api" in item.keywords:
            item.add_marker(skip_api)

        # Skip slow tests in CI
        if "slow" in item.keywords and os.getenv("CI") == "true":
            item.add_marker(pytest.mark.skip(reason="Slow test skipped in CI"))
