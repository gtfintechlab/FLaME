# FLaME (Financial Language Understanding Evaluation)

## Project Setup

### [Basic] Creating and Activating the Virtual Environment

To create the virtual environment in the project root and install the required packages, follow these steps:

1. **Create the virtual environment**:

   ```sh
   python -m venv .venv
   ```
2. **Activate the virtual environment**:

   - On Windows:
     ```sh
     .\.venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```sh
     source .venv/bin/activate
     ```
3. **Install the required packages**:

   ```sh
   pip install -r requirements.txt
   ```

### [Normal] Development Setup with `uv`

For active development, you can use **uv (pip-tools)** to sync dependencies and install `flame` in editable mode:

```sh
pip install pip-tools uv
uv pip compile pyproject.toml -o requirements.txt
uv sync
```

This ensures all packages—including the local `flame` code—are installed in your venv and that code changes are picked up automatically.

**(Recommended) Use uv (pip-tools) for seamless dependency sync and editable package install**:

```sh
pip install pip-tools uv
uv pip compile pyproject.toml -o requirements.txt
uv sync
```

This installs all dependencies and the local `flame` package in editable mode automatically.

### Installing FLaME

After activating your virtual environment, install FLaME in editable mode:

```bash
pip install -e .
```

This creates a link to the local `src/` folder, so changes to the code are picked up automatically without re-installing.

To re-install or upgrade after adding new dependencies:

```bash
pip install --upgrade -e .
```

You can also add the editable install to `requirements.txt` by adding `-e .` at the top, ensuring `pip install -r requirements.txt` will include it.

### Using uv (pip-tools) for dependency sync

If you use `uv` (pip-tools) to manage your environment, you can declare the local package as a path dependency in `pyproject.toml`:

```toml
[project]
dependencies = [
  "flame @ file://./",  # local editable install
  # other deps...
]
```

**Windows users**: relative paths may fail. Use an absolute file URI in `pyproject.toml`:

```toml
dependencies = [
  "flame @ file:///C:/FLaME",  # adjust to your path
  # other deps...
]
```

Then `uv sync` will install the local package.

Confirm by running tests inside the venv -- activate it and use:

```sh
python -m pytest -q
```

Then regenerate and install your lock file without a separate install step:

```bash
# Generate requirements.txt including local flame
uv pip compile pyproject.toml -o requirements.txt
# Sync the venv (installs local flame and all deps)
uv sync
```

### API keys

To configure your API keys, follow these steps:

1. **Create a `.env` file**:

   - You can create a new `.env` file in the project root directory **OR** copy the provided `.env.sample` file and rename it to `.env`.
2. **Modify the `.env` file**:

   - Open the `.env` file in a text editor.
   - Add your API keys in the following format:
     ```
     API_KEY_NAME=your_api_key_value
     ```
   - Replace `API_KEY_NAME` with the actual name of the API key and `your_api_key_value` with your actual API key.
   - Make sure to add all API keys relevant to the models you want to call.
3. **Save the `.env` file**:

   - Ensure the file is saved in the project root directory.

Example:

```
HUGGINGFACEHUB_API_TOKEN=foo
TOGETHER_API_KEY=bar
OPENAI_API_KEY=buzz
ANTHROPIC_API_KEY=buzz
```

## Project Repository

This repository is organized into three primary components: Data Management, Inference Pipeline, and Instructions for Running the Code.

---

### 1. Data Management

The `data` folder in this repository contains multiple subfolders, each representing a different dataset. For each dataset, there is a corresponding script named `huggify_{dataset}.py`. These scripts are designed to upload the raw data for each dataset to the [gtfintech lab&#39;s repository on Hugging Face](https://huggingface.co/gtfintechlab).

#### How to Use the Data Upload Script

For each dataset, you can find its respective script inside its corresponding folder. The script reads raw data files (in CSV/JSON/other formats), processes them, and uploads them to Hugging Face using the Hugging Face API.

- Example directory structure:

```bash
data/ 
    ├── DatasetA/ 
    │ └── huggify_DatasetA.py 
    ├── DatasetB/ 
    │ └── huggify_DatasetB.py
```

To upload a dataset, simply run the respective script:

python3 data/{dataset_name}/huggify_{dataset_name}.py

### 2. Inference Pipeline

The main entry point for the inference process is src/together/inference.py. This script serves as the core orchestrator for running inference tasks on different datasets. It manages the API calls, model loading, and task-specific configurations.

Dataset-Specific Inference Scripts
Each dataset has a dedicated inference script located under src/together/{dataset_name}_inference.py. These scripts contain task-specific inference logic and handle the input/output formats for that particular dataset.

```bash
src/together/
  ├── DatasetA_inference.py
  ├── DatasetB_inference.py
  ├── prompts.py
  └── inference.py
```

Prompts
The file src/together/prompts.py holds various zero-shot prompts that are used for each dataset during inference. These prompts guide the model during the prediction phase.

### 3. Running the FLaME pipeline

Use the unified `main.py` entrypoint to run one or more tasks:

#### Multi-Task Execution

Run multiple tasks with a single command:

```bash
# Using YAML configuration
uv run python main.py --config configs/default.yaml

# Using command-line arguments  
uv run python main.py --config configs/default.yaml --tasks fomc numclaim fnxl --mode inference

# Override specific parameters
uv run python main.py --config configs/default.yaml --tasks fomc numclaim --max_tokens 256 --temperature 0.1
```

#### Task-Specific Configuration

You can specify task-specific parameters in YAML:

```yaml
# configs/multi_task_config.yaml
model: "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
tasks:
  - fomc
  - numclaim
max_tokens: 128
temperature: 0.0

# Task-specific overrides
task_config:
  fomc:
    max_tokens: 256
    temperature: 0.1
  numclaim:
    batch_size: 20
```

#### List Available Tasks

View all available tasks:

```bash
uv run python main.py list-tasks
```

#### Command Options:

- `--config`: Path to your YAML config file (can include `tasks: [task1,task2]`).
- `--mode`: `inference` or `evaluate`.
- `--tasks`: Space-separated list of task names to run.
- `--file_name`: (evaluate only) Path to the inference CSV file.
- `--model`, `--max_tokens`, `--temperature`, `--top_p`, `--top_k`, `--repetition_penalty`, `--batch_size`, `--prompt_format`: Model and generation parameters.

See `docs/multi_task_guide.md` for comprehensive multi-task execution documentation.

## Output Structure

FLaME uses a sophisticated folder hierarchy and filename scheme for organizing results:

### Directory Structure

```
results/
└── {task_slug}/
    └── {provider}/
        └── {model_family}/ (optional)
            └── {model_slug}__{task_slug}__r{run:02d}__{yyyymmdd}__{uuid8}.csv

evaluations/
└── {task_slug}/
    └── {provider}/
        └── {model_family}/ (optional)
            ├── {model_slug}__{task_slug}__r{run:02d}__{yyyymmdd}__{uuid8}.csv
            └── {model_slug}__{task_slug}__r{run:02d}__{yyyymmdd}__{uuid8}_metrics.csv
```

### Filename Template

`{model_slug}__{task_slug}__r{run:02d}__{yyyymmdd}__{uuid8}{suffix}.csv`

Where:
- `model_slug`: Normalized model name (e.g., "llama-4-scout-17b-16e-instruct")
- `task_slug`: Task name (e.g., "causal_classification")  
- `run`: Zero-padded run number (e.g., "01", "02")
- `yyyymmdd`: ISO date string (e.g., "20250524")
- `uuid8`: First 8 characters of UUID for uniqueness
- `suffix`: "" for inference results, "_metrics" for evaluation metrics

### Example Paths

```bash
# Inference result
results/fomc/together_ai/meta-llama/llama-4-scout-17b-16e-instruct__fomc__r01__20250524__a1b2c3d4.csv

# Evaluation results  
evaluations/fomc/together_ai/meta-llama/llama-4-scout-17b-16e-instruct__fomc__r01__20250524__e5f6g7h8.csv
evaluations/fomc/together_ai/meta-llama/llama-4-scout-17b-16e-instruct__fomc__r01__20250524__e5f6g7h8_metrics.csv
```

This structure provides:
- Clear separation by task, provider, and model family
- Collision-free filenames with UUID suffixes
- Consistent naming convention across all outputs
- Easy filtering and organization of results
