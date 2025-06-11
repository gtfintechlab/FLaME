"""Pytest fixtures for Ollama integration testing."""

from unittest.mock import Mock

import pytest
import requests
from litellm import completion


def is_ollama_available():
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


@pytest.fixture
def ollama_model():
    """Fixture that provides Ollama model name for tests."""
    return "ollama/qwen2.5:1.5b"


@pytest.fixture
def ollama_config():
    """Fixture that provides Ollama configuration for tests."""
    return {
        "model": "ollama/qwen2.5:1.5b",
        "api_base": "http://localhost:11434",
        "temperature": 0.0,
        "max_tokens": 128,
        "timeout": 30,
    }


@pytest.fixture
def use_ollama_or_mock(request):
    """
    Fixture that uses real Ollama if available, otherwise mocks.

    Usage in tests:
        def test_something(use_ollama_or_mock):
            response = use_ollama_or_mock(
                messages=[{"role": "user", "content": "test"}]
            )
    """
    if is_ollama_available() and not request.config.getoption(
        "--force-mock", default=False
    ):
        # Use real Ollama
        def _completion(**kwargs):
            default_args = {
                "model": "ollama/qwen2.5:1.5b",
                "api_base": "http://localhost:11434",
                "temperature": 0.0,
                "max_tokens": 128,
            }
            default_args.update(kwargs)
            return completion(**default_args)

        return _completion
    else:
        # Use mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Mocked response: 42"
        mock_response.usage = Mock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        mock_response.model = "ollama/qwen2.5:1.5b"

        def _mock_completion(**kwargs):
            return mock_response

        return _mock_completion


@pytest.fixture
def ollama_integration_test(request):
    """
    Marker for tests that require Ollama to be running.
    Skips test if Ollama is not available.
    """
    if not is_ollama_available():
        pytest.skip("Ollama server not available. Start with 'ollama serve'")

    # Provide test info
    return {
        "model": "ollama/qwen2.5:1.5b",
        "api_base": "http://localhost:11434",
        "available": True,
    }


# Pytest configuration hook
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--force-mock",
        action="store_true",
        default=False,
        help="Force use of mocks even if Ollama is available",
    )
    parser.addoption(
        "--ollama-only",
        action="store_true",
        default=False,
        help="Run only tests that use real Ollama (skip if not available)",
    )
