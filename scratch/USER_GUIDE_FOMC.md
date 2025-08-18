# FLAME FOMC Task User Guide

## Quick Start

FLAME is now fully integrated with BenchForge infrastructure! You can run FOMC (Federal Open Market Committee) sentiment classification with simple commands.

## Installation

```bash
# Install FLAME
cd /home/gmatlin/Codespace/FLAME
uv pip install -e .

# Install BenchForge (submodule)
cd benchforge
uv pip install -e .
cd ..
```

## Running FOMC Task

### 1. Check Status

First, verify everything is working:

```bash
uv run flame --mode status
```

Expected output:
```
============================================================
FLAME-BenchForge Integration Status
============================================================
BenchForge Available: True
BenchForge Version: 0.4.0
Registered Tasks: 2
Available tasks:
  - fomc
  - fpb
============================================================
```

### 2. Run Inference

Run FOMC inference with the default model:

```bash
uv run flame --mode inference \
  --task fomc \
  --model "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct" \
  --batch_size 20 \
  --max_tokens 10 \
  --temperature 0.0
```

Or run with a limited number of samples for testing:

```bash
uv run flame --mode inference \
  --task fomc \
  --model "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct" \
  --num_samples 10 \
  --batch_size 5
```

### 3. Run Evaluation

After inference completes, evaluate the results:

```bash
# Find your results file
ls results/fomc/

# Run evaluation
uv run flame --mode evaluate \
  --task fomc \
  --file_name "results/fomc/fomc_20250816_XXXXXX.csv" \
  --metrics accuracy f1_macro
```

## Command Options

### Inference Options

| Option | Default | Description |
|--------|---------|-------------|
| `--task` | required | Task name (fomc) |
| `--model` | required | Model identifier |
| `--batch_size` | 10 | Batch size for processing |
| `--max_tokens` | 256 | Maximum tokens to generate |
| `--temperature` | 0.0 | Sampling temperature |
| `--top_p` | 1.0 | Nucleus sampling parameter |
| `--num_samples` | None | Number of samples (None = all) |
| `--prompt_format` | zero_shot | Prompt format (zero_shot, few_shot, chain_of_thought) |
| `--split` | test | Dataset split to use |

### Evaluation Options

| Option | Description |
|--------|-------------|
| `--file_name` | Path to results CSV file |
| `--metrics` | List of metrics (accuracy, f1_macro, etc.) |
| `--output_dir` | Directory for evaluation results |

## FOMC Task Details

### Dataset
- **Source**: HuggingFace - `gtfintechlab/fomc_communication`
- **Size**: 496 test samples
- **Task**: Classify FOMC statements as HAWKISH, DOVISH, or NEUTRAL

### Labels
- **HAWKISH**: Indicates tightening monetary policy (raising rates, reducing stimulus)
- **DOVISH**: Indicates loosening monetary policy (lowering rates, increasing stimulus)
- **NEUTRAL**: Indicates maintaining current policy stance

### Prompt Formats

#### Zero-Shot (default)
Simple classification without examples.

#### Few-Shot
Includes 3 examples before the target text.

#### Chain-of-Thought
Step-by-step reasoning approach.

## Output Files

### Inference Results
- Location: `results/fomc/`
- Format: CSV with columns:
  - `index`: Sample index
  - `input`: Original text
  - `prompt`: Generated prompt
  - `raw_response`: Model's raw output
  - `extracted_response`: Extracted label
  - `ground_truth`: True label
  - `model`: Model used
  - `timestamp`: When processed

### Evaluation Results
- Location: `evaluations/fomc/`
- Files:
  - `eval_fomc_*.json`: Full evaluation results
  - `eval_fomc_*_metrics.csv`: Metrics summary

## Environment Variables

Set these in your `.env` file:

```bash
# Required for inference
TOGETHER_API_KEY=your_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_token_here

# Optional
OPENAI_API_KEY=your_key_here  # If using OpenAI models
```

## Troubleshooting

### API Key Issues
```bash
# Check if API keys are set
env | grep API_KEY
```

### Module Not Found
```bash
# Reinstall packages
uv pip install -e .
cd benchforge && uv pip install -e . && cd ..
```

### Out of Memory
- Reduce `--batch_size` (try 5 or 10)
- Use `--num_samples` to limit dataset size

## Advanced Usage

### Custom Configuration File

Create a YAML config file:

```yaml
# config.yaml
task: fomc
model: "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
batch_size: 20
max_tokens: 10
temperature: 0.0
prompt_format: zero_shot
```

Run with config:
```bash
uv run flame --mode inference --config config.yaml
```

### Verbose Output

For debugging:
```bash
uv run flame --mode inference --task fomc --model <model> --verbose
```

### Quiet Mode

For minimal output:
```bash
uv run flame --mode inference --task fomc --model <model> --quiet
```

## Performance Tips

1. **Batch Size**: Start with 10-20, adjust based on API limits
2. **Max Tokens**: FOMC only needs 10 tokens (single word output)
3. **Temperature**: Use 0.0 for deterministic results
4. **Caching**: Results are cached in BenchForge for faster re-runs

## Next Steps

- Try different models by changing the `--model` parameter
- Experiment with `--prompt_format few_shot` for potentially better accuracy
- Run other FLAME tasks like `fpb` (Financial PhraseBank)

## Support

For issues or questions:
1. Check the test script: `scratch/test_fomc_e2e.py`
2. Review logs in verbose mode
3. Verify BenchForge status with `uv run flame --mode status`