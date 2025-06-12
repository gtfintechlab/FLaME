#!/bin/bash
# Script to run tests in groups to avoid isolation issues
# This prevents test failures caused by module import side effects

set -e  # Exit on error

echo "Running FLaME tests in isolated groups..."

# Run unit tests first (most isolated)
echo "1. Running unit tests..."
uv run pytest -m unit -v

# Run prompt tests (lightweight, no heavy imports)
echo "2. Running prompt tests..."
uv run pytest -m prompts -v

# Run multi-task tests
echo "3. Running multi-task tests..."
uv run pytest -m multi_task -v

# Run integration tests
echo "4. Running integration tests..."
uv run pytest -m integration -v

# Run module tests last (these have the most import side effects)
echo "5. Running module tests..."
uv run pytest -m modules -v

echo "All test groups completed successfully!"