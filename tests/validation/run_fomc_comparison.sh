#!/bin/bash
#
# FOMC Implementation Comparison Test Script
# ===========================================
# This script runs FOMC with both native FLAME and BenchForge implementations
# to validate feature parity for Phase 1 completion.
#

set -e  # Exit on error

echo "================================================"
echo "FOMC Feature Parity Validation - Phase 1"
echo "================================================"
echo ""

# Configuration
MODEL="together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
TEST_SAMPLES=5
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="validation_results_${TIMESTAMP}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "Configuration:"
echo "  Model: ${MODEL}"
echo "  Test Samples: ${TEST_SAMPLES}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo ""

# Step 1: Test Native FLAME Implementation
echo "================================================"
echo "Step 1: Testing Native FLAME Implementation"
echo "================================================"
echo ""

# Check if native implementation exists
if [ -f "src/flame/code/fomc/fomc_inference.py" ]; then
    echo "✅ Native FLAME implementation found"
    
    # Test import and basic functionality
    python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from flame.code.fomc.fomc_inference import fomc_inference
    from flame.code.prompts.registry import get_prompt, PromptFormat
    from flame.code.fomc.fomc_evaluate import map_label_to_number
    
    # Test prompt generation
    prompt_func = get_prompt('fomc', PromptFormat.ZERO_SHOT)
    test_prompt = prompt_func('The Committee decided to raise rates.')
    print('✅ Native: Prompt generation working')
    
    # Test label mapping
    assert map_label_to_number('HAWKISH') == 1
    assert map_label_to_number('DOVISH') == 0
    assert map_label_to_number('NEUTRAL') == 2
    print('✅ Native: Label mapping working')
    
    print('✅ Native FLAME: All basic tests passed')
except Exception as e:
    print(f'❌ Native FLAME: Test failed - {e}')
    sys.exit(1)
" 2>&1 | tee ${OUTPUT_DIR}/native_test.log
else
    echo "❌ Native FLAME implementation not found"
    exit 1
fi

echo ""

# Step 2: Test BenchForge Implementation
echo "================================================"
echo "Step 2: Testing BenchForge Implementation"
echo "================================================"
echo ""

# Check if BenchForge is available
if [ -d "benchforge" ]; then
    echo "✅ BenchForge directory found"
    
    # Test import and basic functionality
    python3 -c "
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'benchforge')
try:
    from flame.benchforge import BENCHFORGE_AVAILABLE
    
    if not BENCHFORGE_AVAILABLE:
        print('❌ BenchForge not properly installed')
        sys.exit(1)
    
    from flame.tasks.fomc import FOMCTask
    from flame.benchforge import FLAMEConfig, PromptFormat
    
    # Create task instance
    config = FLAMEConfig(
        name='fomc',
        dataset='fomc',
        prompt_format=PromptFormat.ZERO_SHOT,
        text_field='sentence',
        label_field='label',
        valid_labels=['HAWKISH', 'DOVISH', 'NEUTRAL']
    )
    
    task = FOMCTask(config)
    print('✅ BenchForge: Task initialization working')
    
    # Test prompt generation
    test_sample = {'sentence': 'The Committee decided to raise rates.'}
    prompt = task.create_prompt(test_sample, PromptFormat.ZERO_SHOT)
    print('✅ BenchForge: Prompt generation working')
    
    # Test extraction
    extracted = task.extract_label_from_response('HAWKISH', use_llm_fallback=False)
    assert extracted == 'HAWKISH'
    mapped = task.map_label_to_number(extracted)
    assert mapped == 1
    print('✅ BenchForge: Extraction and mapping working')
    
    print('✅ BenchForge: All basic tests passed')
except Exception as e:
    print(f'❌ BenchForge: Test failed - {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee ${OUTPUT_DIR}/benchforge_test.log
else
    echo "❌ BenchForge not found. Please initialize with: git submodule update --init"
    exit 1
fi

echo ""

# Step 3: Run Comparison Tests
echo "================================================"
echo "Step 3: Running Comparison Tests"
echo "================================================"
echo ""

# Run the validation script
echo "Running comprehensive validation..."
python3 tests/validation/phase1_fomc_validation.py \
    --samples ${TEST_SAMPLES} \
    --output ${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/validation.log

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✅ PHASE 1 VALIDATION COMPLETE"
    echo "================================================"
    echo ""
    echo "Both implementations are working correctly with feature parity."
    echo "Results saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Next Steps for Phase 2:"
    echo "1. Review validation report in ${OUTPUT_DIR}/validation_summary_*.md"
    echo "2. Run actual inference tests with small dataset"
    echo "3. Begin gradual migration using feature flags"
    echo ""
else
    echo ""
    echo "================================================"
    echo "❌ VALIDATION FAILED"
    echo "================================================"
    echo ""
    echo "Please review the logs in ${OUTPUT_DIR} for details."
    echo ""
    exit 1
fi