#!/bin/bash
"""Run FinEntity implementation comparison between BenchForge and Native FLAME."""

set -e  # Exit on error

# Set working directory to FLAME root
cd "$(dirname "$0")/../../"

echo "=========================================="
echo "FinEntity Implementation Comparison"
echo "=========================================="

# Check Python environment
echo "Checking Python environment..."
if ! command -v python &> /dev/null; then
    echo "❌ Python not found in PATH"
    exit 1
fi

python --version
echo "Current directory: $(pwd)"

# Check API keys
echo "Checking API keys..."
if [ -z "$TOGETHERAI_API_KEY" ]; then
    if [ -f ".env" ]; then
        echo "Loading .env file..."
        export $(grep -v '^#' .env | xargs)
    else
        echo "⚠️ TOGETHERAI_API_KEY not found and no .env file"
        echo "Please set the TOGETHERAI_API_KEY environment variable"
        exit 1
    fi
fi

if [ -n "$TOGETHERAI_API_KEY" ]; then
    echo "✓ TogetherAI API key found"
else
    echo "❌ TOGETHERAI_API_KEY not found"
    exit 1
fi

# Default parameters
MODEL="together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
LIMIT=""
BATCH_SIZE=20

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model MODEL       Model to use (default: $MODEL)"
            echo "  --limit N           Limit to N samples (default: all)"
            echo "  --batch-size N      Batch size for processing (default: $BATCH_SIZE)"
            echo "  --help              Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Batch size: $BATCH_SIZE"
if [ -n "$LIMIT" ]; then
    echo "  Sample limit: $LIMIT"
else
    echo "  Sample limit: All samples"
fi

# Run the comparison
echo ""
echo "Starting FinEntity implementation comparison..."
echo "=========================================="

python benchforge/tests/test_finentity_comparison.py \
    --model "$MODEL" \
    $LIMIT \
    --save

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "=========================================="

# Show result files
echo "Result files:"
find results/finentity_comparison -name "*.csv" -o -name "*.json" | sort | tail -6

echo ""
echo "To view results:"
echo "  cd results/finentity_comparison"
echo "  ls -la"