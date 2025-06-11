# FLaME Scripts

This directory contains utility scripts for development and validation of the FLaME framework.

## Directory Structure

### `/ollama`
Scripts for testing and validating tasks using Ollama for local LLM inference:
- `test_ollama_connection.py` - Tests basic Ollama-LiteLLM integration
- `test_phase2_with_ollama.py` - Automated Phase 2 task testing with Ollama

### `/validation`
Scripts for validation and status checking:
- `check_phase2_status.py` - Checks the status of Phase 2 validation fixes
- `test_convfinqa_small.py` - Quick test for convfinqa task

## Usage

All scripts should be run from the project root directory using `uv`:

```bash
# Test Ollama connection
uv run python scripts/ollama/test_ollama_connection.py

# Check Phase 2 status
uv run python scripts/validation/check_phase2_status.py

# Run Phase 2 validation with Ollama
uv run python scripts/validation/test_phase2_with_ollama.py
```

## Notes
- These scripts are for development and testing purposes
- Production runs should use the main.py entry point
- Ollama must be running locally for Ollama scripts to work