# Parallel Testing Plan for FOMC Model Comparison

## Objective
Run 50-sample tests on 5 models using both native FLAME and BenchForge methods in parallel to maximize efficiency and enable comprehensive comparison.

## Test Configuration

### Models to Test
1. **Llama-3.2-3B** - `together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo` (Small/Fast)
2. **Mistral-7B** - `together_ai/mistralai/Mistral-7B-Instruct-v0.3` (Small-Medium)
3. **Llama-3.1-8B** - `together_ai/meta-llama/Llama-3.1-8B-Instruct-Turbo` (Medium)
4. **Mistral-24B** - `together_ai/mistralai/Mistral-Small-24B-Instruct-2501` (Medium-Large)
5. **Llama-3.3-70B** - `together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo` (Large)

### Test Parameters
- **Sample Size**: 50 samples per model
- **Batch Size**: 10 (for efficient API usage)
- **Total Tests**: 10 (5 models × 2 methods)
- **Estimated Time**: ~15-20 minutes total with parallelization

## Parallel Execution Strategy

### Phase 1: Setup and Validation
1. Verify API key is set
2. Load first 50 samples from FOMC test dataset
3. Create output directories for results
4. Initialize logging for each parallel process

### Phase 2: Parallel Model Testing

#### Split Strategy
Instead of running complete comparison sequentially, split into independent tasks:

**Task Group A - Native FLAME (5 parallel tasks):**
- Task A1: FLAME + Llama-3.2-3B
- Task A2: FLAME + Mistral-7B  
- Task A3: FLAME + Llama-3.1-8B
- Task A4: FLAME + Mistral-24B
- Task A5: FLAME + Llama-3.3-70B

**Task Group B - BenchForge (5 parallel tasks):**
- Task B1: BenchForge + Llama-3.2-3B
- Task B2: BenchForge + Mistral-7B
- Task B3: BenchForge + Llama-3.1-8B
- Task B4: BenchForge + Mistral-24B
- Task B5: BenchForge + Llama-3.3-70B

### Phase 3: Result Aggregation
1. Collect all 10 result files
2. Compare matching model pairs
3. Generate comparison metrics
4. Create summary report

## Implementation Files

### 1. `run_parallel_tests.py`
Main orchestrator that:
- Spawns parallel processes for each task
- Monitors progress
- Aggregates results

### 2. `run_single_flame_test.py`
Standalone script for native FLAME testing:
- Takes model name as argument
- Runs 50 samples
- Saves results to `results/flame/{model_name}_50samples.csv`
- Saves metrics to `results/flame/{model_name}_metrics.csv`

### 3. `run_single_benchforge_test.py`
Standalone script for BenchForge testing:
- Takes model name as argument
- Runs 50 samples  
- Saves results to `results/benchforge/{model_name}_50samples.csv`
- Saves metrics to `results/benchforge/{model_name}_metrics.csv`

### 4. `aggregate_results.py`
Post-processing script that:
- Loads all result files
- Compares FLAME vs BenchForge for each model
- Generates comparison report

## Expected Output Structure

```
results/
├── flame/
│   ├── llama-3.2-3b_50samples.csv
│   ├── llama-3.2-3b_metrics.csv
│   ├── mistral-7b_50samples.csv
│   ├── mistral-7b_metrics.csv
│   └── ... (8 more files)
├── benchforge/
│   ├── llama-3.2-3b_50samples.csv
│   ├── llama-3.2-3b_metrics.csv
│   ├── mistral-7b_50samples.csv
│   ├── mistral-7b_metrics.csv
│   └── ... (8 more files)
└── comparison/
    ├── model_comparison_summary.json
    ├── accuracy_comparison.csv
    ├── extraction_rates.csv
    └── detailed_report.md
```

## Metrics to Compare

### Primary Metrics
1. **Accuracy**: Overall classification accuracy
2. **F1 Score**: Weighted F1 score
3. **Extraction Rate**: Percentage of successful label extractions
4. **Processing Time**: Time per sample

### Secondary Metrics
1. **Per-class Precision/Recall**: For DOVISH, HAWKISH, NEUTRAL
2. **Consistency**: Agreement between methods on same samples
3. **Error Patterns**: Common failure cases

### Comparison Analysis
1. **Method Agreement**: % of samples where both methods extracted same label
2. **Accuracy Delta**: Difference in accuracy between methods
3. **Extraction Success**: Which method extracts more labels successfully
4. **Performance**: Speed comparison

## Error Handling

### API Rate Limiting
- Implement exponential backoff
- Distribute requests across time
- Use 0.5s delay between batches

### Model Availability
- Check model availability before starting
- Use fallback model if primary unavailable
- Log all model substitutions

### Partial Failures
- Save progress after each batch
- Allow resume from checkpoint
- Mark failed samples clearly

## Execution Commands

### Step 1: Run All Tests in Parallel
```bash
uv run python benchforge/run_parallel_tests.py --samples 50 --batch-size 10
```

### Step 2: Monitor Progress
```bash
# Watch log files in real-time
tail -f results/logs/*.log
```

### Step 3: Generate Comparison Report
```bash
uv run python benchforge/aggregate_results.py --output results/comparison/
```

## Alternative: Sequential with Progress Saving

If parallel execution has issues, run sequentially with checkpointing:

```bash
# Run each model-method pair individually
for model in llama-3.2-3b mistral-7b llama-3.1-8b mistral-24b llama-3.3-70b; do
    uv run python benchforge/run_single_flame_test.py --model $model --samples 50 &
    uv run python benchforge/run_single_benchforge_test.py --model $model --samples 50 &
    wait
done
```

## Success Criteria

1. **All 10 tests complete** (5 models × 2 methods)
2. **Each test processes 50 samples**
3. **Results saved in standardized format**
4. **Comparison report generated**
5. **BenchForge confirmed as superset** (has all FLAME features + more)

## Post-Test Analysis

After tests complete, analyze:

1. **Model Performance Ranking**: Which models perform best on FOMC?
2. **Method Comparison**: Is BenchForge consistently equal or better?
3. **Extraction Strategy Success**: How often does each extraction strategy succeed?
4. **Error Analysis**: Common failure patterns across models/methods
5. **Recommendations**: Which model-method combination is optimal?

## Risk Mitigation

1. **API Key Issues**: Verify key before starting all tests
2. **Memory Issues**: Run in batches, clear memory between models
3. **Network Failures**: Implement retry logic with backoff
4. **Data Consistency**: Use same 50 samples for all tests
5. **Result Validation**: Verify output files are complete before aggregation