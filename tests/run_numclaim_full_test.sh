#!/bin/bash
# Run NumClaim full dataset test with TogetherAI

echo "=================================="
echo "NumClaim Full Dataset Test"
echo "=================================="

# Set batch size (adjust based on API rate limits)
export BATCH_SIZE=10

# Ensure API key is set
if [ -z "$TOGETHER_API_KEY" ] && [ -z "$TOGETHERAI_API_KEY" ]; then
    echo "❌ Error: TOGETHER_API_KEY not set"
    echo "Please set: export TOGETHER_API_KEY='your-key'"
    exit 1
fi

# Navigate to the test directory
cd /home/gmatlin/Codespace/FLAME/benchforge/tests

# Run the full dataset test
echo "Starting full dataset test with batch size: $BATCH_SIZE"
echo "This will test 537 samples from the NumClaim test set"
echo ""

# Use uv to run with correct environment
uv run python test_numclaim_full_dataset.py

echo ""
echo "Test completed!"