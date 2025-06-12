"""Integration test configuration."""

# Import ollama fixtures
from tests.fixtures.ollama_fixtures import (
    ollama_config,
    ollama_integration_test,
    ollama_model,
    use_ollama_or_mock,
)

__all__ = [
    "ollama_model",
    "ollama_config",
    "use_ollama_or_mock",
    "ollama_integration_test",
]
