# Sample Size Limiting Feature

## Overview

The FLaME framework now supports limiting the number of samples used from each dataset during inference. This feature is useful for:

1. Quick testing and development with cheaper API costs
2. Rapid prototyping and validation
3. Running smoke tests on new models
4. Limited budget scenarios

## Supported Tasks

All tasks in the FLaME framework now support the sample_size parameter:

- banking77
- bizbench
- causal_classification
- causal_detection
- convfinqa
- econlogicqa
- ectsum
- edtsum
- finbench
- finentity
- finer
- finqa
- finred
- fiqa_task1
- fiqa_task2
- fnxl
- fomc
- fpb
- headlines
- mmlu
- numclaim
- refind
- subjectiveqa
- tatqa

## Usage

### CLI Arguments

You can specify the number of samples to use with the `--sample_size` argument:

```bash
uv run python main.py --mode inference --model "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct" --tasks fpb fomc --sample_size 10
```

This will limit both the FPB and FOMC tasks to use only the first 10 samples from their respective test datasets.

### YAML Configuration

You can also set the sample size in your YAML configuration file:

```yaml
model: "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
tasks:
  - fomc
  - fpb
sample_size: 10
max_tokens: 128
temperature: 0.7
```

### CLI Override

CLI arguments take precedence over YAML configuration:

```bash
# This will use 5 samples even if the config file specifies 10
uv run python main.py --config configs/default.yaml --mode inference --sample_size 5
```

## Behavior

1. **Sequential Selection**: Samples are selected sequentially from the beginning of the dataset
2. **Exceeding Dataset Size**: If `sample_size` exceeds the actual dataset size, all available samples are used
3. **None or Unspecified**: When `sample_size` is not specified or is None, all samples are processed (default behavior)
4. **Zero Value**: Setting `sample_size` to 0 will process no samples

## Implementation Details

The feature is implemented in the task-specific inference functions. For example, in FPB inference:

```python
test_data = dataset["test"]

# Apply sample size limit if specified
if hasattr(args, 'sample_size') and args.sample_size is not None:
    test_data = test_data.select(range(min(args.sample_size, len(test_data))))
    logger.info(f"Limited dataset to {len(test_data)} samples")
```

## Backward Compatibility

The implementation maintains full backward compatibility:

- If the `sample_size` attribute is not present in the args object, the system uses all samples
- Existing scripts and configurations continue to work without modification
- The feature is opt-in and doesn't affect existing behavior when not specified

## Testing

Comprehensive tests have been added to ensure:

1. Correct parsing of CLI and YAML arguments
2. Proper dataset limiting behavior
3. Backward compatibility
4. Error handling for invalid inputs
5. Integration with the existing inference pipeline

Run the tests with:

```bash
uv run pytest tests/test_sample_size.py -v
uv run pytest tests/test_dataset_sampling.py -v
uv run pytest tests/test_backward_compatibility.py -v
```

## Examples

### Quick Development Test

Test with just 3 samples to verify your setup:

```bash
uv run python main.py --config configs/default.yaml --mode inference --sample_size 3
```

### Budget-Conscious Run

Process 50 samples to get meaningful results while controlling costs:

```bash
uv run python main.py --mode inference --model "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct" --tasks fpb fomc finentity --sample_size 50
```

### Full Dataset Run

Omit the `sample_size` argument to process all available samples:

```bash
uv run python main.py --config configs/default.yaml --mode inference
```