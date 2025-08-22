#!/bin/bash
"""Run FinEntity with both Native FLAME and BenchForge implementations separately."""

set -e  # Exit on error

# Set working directory to FLAME root
cd "$(dirname "$0")/../../"

echo "=========================================="
echo "FinEntity Separate Implementation Runs"
echo "=========================================="

# Configuration
MODEL="together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
BATCH_SIZE=20
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Check API keys
if [ -f ".env" ]; then
    echo "Loading .env file..."
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$TOGETHERAI_API_KEY" ]; then
    echo "❌ TOGETHERAI_API_KEY not found"
    echo "Please set the TOGETHERAI_API_KEY environment variable"
    exit 1
fi

echo "✓ TogetherAI API key found"
echo "Model: $MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Timestamp: $TIMESTAMP"

# Create results directory
mkdir -p results/finentity

echo ""
echo "=========================================="
echo "1. Running Native FLAME FinEntity"
echo "=========================================="

python main.py \
    --mode inference \
    --tasks finentity \
    --model "$MODEL" \
    --batch_size $BATCH_SIZE \
    --temperature 0.0 \
    --max_tokens 512 \
    --prompt_format zero_shot \
    --use-native

# Find the most recent native result
NATIVE_RESULT=$(find results -name "*finentity*" -type f -exec stat -c "%Y %n" {} + | sort -nr | head -1 | cut -d' ' -f2-)
if [ -n "$NATIVE_RESULT" ]; then
    echo "✓ Native FLAME result saved to: $NATIVE_RESULT"
    # Copy to our comparison directory with clear naming
    cp "$NATIVE_RESULT" "results/finentity/native_flame_finentity_${TIMESTAMP}.csv"
    echo "✓ Copied to: results/finentity/native_flame_finentity_${TIMESTAMP}.csv"
else
    echo "⚠️ Native FLAME result file not found"
fi

echo ""
echo "=========================================="
echo "2. Running BenchForge FinEntity"
echo "=========================================="

python main.py \
    --mode inference \
    --tasks finentity \
    --model "$MODEL" \
    --batch_size $BATCH_SIZE \
    --temperature 0.0 \
    --max_tokens 512 \
    --prompt_format zero_shot \
    --use-benchforge

# Find the most recent benchforge result
BENCHFORGE_RESULT=$(find results -name "*finentity*" -type f -exec stat -c "%Y %n" {} + | sort -nr | head -1 | cut -d' ' -f2-)
if [ -n "$BENCHFORGE_RESULT" ]; then
    echo "✓ BenchForge result saved to: $BENCHFORGE_RESULT"
    # Copy to our comparison directory with clear naming
    cp "$BENCHFORGE_RESULT" "results/finentity/benchforge_finentity_${TIMESTAMP}.csv"
    echo "✓ Copied to: results/finentity/benchforge_finentity_${TIMESTAMP}.csv"
else
    echo "⚠️ BenchForge result file not found"
fi

echo ""
echo "=========================================="
echo "3. Running Evaluations"
echo "=========================================="

if [ -f "results/finentity/native_flame_finentity_${TIMESTAMP}.csv" ]; then
    echo "Evaluating Native FLAME results..."
    python main.py \
        --mode evaluate \
        --tasks finentity \
        --file_name "results/finentity/native_flame_finentity_${TIMESTAMP}.csv" \
        --use-native
    
    # Find and copy the evaluation result
    EVAL_RESULT=$(find evaluations -name "*finentity*" -type f -exec stat -c "%Y %n" {} + | sort -nr | head -1 | cut -d' ' -f2-)
    if [ -n "$EVAL_RESULT" ]; then
        cp "$EVAL_RESULT" "results/finentity/native_flame_finentity_${TIMESTAMP}_evaluation.csv"
        echo "✓ Native FLAME evaluation saved"
    fi
fi

if [ -f "results/finentity/benchforge_finentity_${TIMESTAMP}.csv" ]; then
    echo "Evaluating BenchForge results..."
    python main.py \
        --mode evaluate \
        --tasks finentity \
        --file_name "results/finentity/benchforge_finentity_${TIMESTAMP}.csv" \
        --use-benchforge
    
    # Find and copy the evaluation result
    EVAL_RESULT=$(find evaluations -name "*finentity*" -type f -exec stat -c "%Y %n" {} + | sort -nr | head -1 | cut -d' ' -f2-)
    if [ -n "$EVAL_RESULT" ]; then
        cp "$EVAL_RESULT" "results/finentity/benchforge_finentity_${TIMESTAMP}_evaluation.csv"
        echo "✓ BenchForge evaluation saved"
    fi
fi

echo ""
echo "=========================================="
echo "Separate runs complete!"
echo "=========================================="

echo "Results directory: results/finentity/"
ls -la results/finentity/ | grep "$TIMESTAMP"

echo ""
echo "To analyze results:"
echo "  cd results/finentity"
echo "  python -c \"import pandas as pd; df1=pd.read_csv('native_flame_finentity_${TIMESTAMP}.csv'); df2=pd.read_csv('benchforge_finentity_${TIMESTAMP}.csv'); print('Native samples:', len(df1)); print('BenchForge samples:', len(df2))\""