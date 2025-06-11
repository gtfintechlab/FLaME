#!/bin/bash
# CI test runner script with proper environment setup

set -e  # Exit on error

echo "Setting up CI environment..."
export CI=true
export PYTEST_RUNNING=1
export HUGGINGFACEHUB_API_TOKEN="mock-token-for-ci"
export FLAME_CONFIG="ci"

# Copy CI conftest to override default
cp tests/conftest_ci.py tests/conftest_ci_override.py

echo "Running tests with CI configuration..."
python -m pytest -c pytest-ci.ini "$@"

# Cleanup
rm -f tests/conftest_ci_override.py

echo "CI tests completed."