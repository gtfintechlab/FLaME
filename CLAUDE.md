# CLAUDE.md - AI Agent Guidelines

This document defines the operational guidelines, tool usage standards, and behavioral rules for AI agents working in shell-based and code development workflows. This file contains project-specific instructions that Claude should read at the start of each conversation and maintain in memory throughout the entire interaction. **IMPORTANT:** Once this file has been read or updated, it MUST be loaded at the beginning of any new conversation to ensure awareness of communication requirements, custom tasks, etc.

## Core Principles and Guidelines

### Default Mode

- Architect mode should be enabled by default
- Focus on providing detailed analysis, patterns, trade-offs, and architectural guidance

### Permissions

- Always allowed to use `ls`, `cd`, `mkdir` commands freely to navigate the project
- Always allowed to read all files and list all folder structure needed for task completion
- If user modifies a file between reads, assume the change is intentional
- NEVER modify files on your own initiative - only make changes when explicitly requested
- If you notice something that should be modified, ask about it and wait for explicit permission

### Code Style Guidelines

- **Files/Components**: PascalCase for components, camelCase for utils/hooks
- **Types**: Strict typing, descriptive generics, no implicit any, named prop interfaces
- **Naming**: Function types use FunctionNameArgs, class options use ClassNameOptions, hook args use UseHookNameArgs, React component props use ComponentNameProps
- **Error Handling**: Custom error classes, i18n error messages, meaningful error types
- **Patterns**: Prefer immutability (const over let), no direct process.env usage, prefer ts-pattern over switch
- **Components**: One component per file, functional components with hooks
- **CSS**: Tailwind with twMerge for class composition
- **Dates**: Use Luxon over native Date objects
- **Type organization**: Don't create `index.ts` files in `types` directories to re-export types, import directly from individual type files
- **Comments**:
  - Use minimal comments and only in English
  - Add comments only when code clarity is insufficient or to explain non-standard solutions (like using `any`) or hard to read / understand code sections
  - Don't use JSDoc style function header comments

### Git Commit Guidelines

Anonymity is important. Never mention names of people, AI, Claude, or any AI assistant names in:
   - Git commit messages
   - Code comments (unless explaining technical logic)
   - Documentation
   - Any user-facing output

### Communications Guidelines 

- Avoid hyperbole, over-complementary language, or "encouraging" tone
- State facts and progress objectively
- Do not overstate completed work or project status
- NEVER suggest or offer staging files with git add commands
- When asking questions, always provide multiple numbered options when appropriate:

  - Format as a numbered list: `1. Option one, 2. Option two, 3. Option three`
  - Example: `1. Yes, continue with the changes, 2. Modify the approach, 3. Stop and cancel the operation`

- When analyzing code for improvement:

  - Present multiple implementation variants as numbered options
  - For each variant, provide at least 3 bullet points explaining the changes, benefits, and tradeoffs
  - Format as: "1. [short exmplanation of variant or shorly Variant]" followed by explanation points

- When implementing code changes:

  - If the change wasn't preceded by an explanation or specific instructions
  - Include within the diff a bulleted list explaining what was changed and why
  - Explicitly note when a solution is opinionated and explain the reasoning

- When completing a task, ask if I want to:
  1. Run task:commit (need to manually stage files first)
  2. Neither (stop here)

### Code Style Consistency

- ALWAYS respect how things are written in the existing project
- DO NOT invent your own approaches or innovations
- STRICTLY follow the existing style of tests, resolvers, functions, and arguments
- Before creating a new file, ALWAYS examine a similar file and follow its style exactly
- If code doesn't include comments, DO NOT add comments
- Use seeded data in tests instead of creating new objects when seeded data exists
- Follow the exact format of error handling, variable naming, and code organization used in similar files
- Never deviate from the established patterns in the codebase

### Code Documentation and Comments

When working with code that contains comments or documentation:

1. Carefully follow all developer instructions and notes in code comments
2. Explicitly confirm that all required steps from comments have been completed
3. Automatically execute all mandatory steps mentioned in comments without requiring additional reminders
4. Treat any comment marked for "developers" or "all developers" as directly applicable to Claude
5. Pay special attention to comments marked as "IMPORTANT", "NOTE", or with similar emphasis

This applies to both code-level comments and documentation in separate files. Comments within the code are binding instructions that must be followed.

### Knowledge Sharing and Persistence

- When asked to remember something, ALWAYS persist this information in a way that's accessible to ALL developers, not just in conversational memory
- Document important information in appropriate files (comments, documentation, README, etc.) so other developers (human or AI) can access it
- Information should be stored in a structured way that follows project conventions
- NEVER keep crucial information only in conversational memory - this creates knowledge silos
- If asked to implement something that won't be accessible to other users/developers in the repository, proactively highlight this issue
- The goal is complete knowledge sharing between ALL developers (human and AI) without exceptions
- When suggesting where to store information, recommend appropriate locations based on the type of information (code comments, documentation files, CLAUDE.md, etc.)

### Commands and Tasks

- Files in the `.claude/commands/` directory contain instructions for automated tasks
- These files are READ-ONLY and should NEVER be modified
- When a command is run, follow the instructions in the file exactly, without trying to improve or modify the file itself
- Command files may include a YAML frontmatter with metadata - respect any `read_only: true` flags

### Path References

- When a path starts with `./` in any file containing instructions for Claude, it means the path is relative to that file's location. Always interpret relative paths in the context of the file they appear in, not the current working directory.

## CLI Tool Standards

### File Operations

**Search & Discovery**
- **Content search**: Use `rg` (ripgrep) exclusively
  ```bash
  rg "pattern" path/to/dir
  rg -i "case_insensitive" .
  rg -t py "import pandas" src/
  ```

- **File/directory search**: Use `fd` for fast filesystem queries
  ```bash
  fd "\.py$" src/
  fd -t d "test" .
  ```

**File Viewing**
- **Interactive viewing**: Use `bat` for syntax-highlighted display
  ```bash
  bat config.yaml
  bat -n script.py  # with line numbers
  ```

- **Piped output**: Use `cat` when feeding to other tools
  ```bash
  cat data.json | jq '.results[]'
  cat script.sh | shellcheck -
  ```

### Directory & Version Control

- **Directory listings**: Use `eza` instead of `ls`
  ```bash
  eza -lah --git
  eza --tree --level=2
  ```

- **Git diffs**: Use `delta` as the diff pager
  ```bash
  git diff --color | delta
  git show HEAD~1 | delta
  ```

### Data Processing

- **JSON**: Use `jq` for all JSON operations
  ```bash
  jq '.items[] | select(.active == true)' data.json
  jq -r '.users[].email' users.json
  ```

- **YAML**: Use `yq` for YAML manipulation
  ```bash
  yq eval '.services.*.port' docker-compose.yml
  yq eval '.version = "3.9"' -i config.yaml
  ```


### Build & Test Commands

#### Using uv (recommended)
- Install dependencies: `uv pip install --system -e .`
- Install dev dependencies: `uv pip install --system -e ".[dev]"`
- Update lock file: `uv pip compile --system pyproject.toml -o uv.lock`
- Install from lock file: `uv pip sync --system uv.lock`

#### Using pip (alternative)
- Install dependencies: `pip install -e .`
- Install dev dependencies: `pip install -e ".[dev]"`

#### Testing and linting
- Run tests: `pytest`
- Run single test: `pytest tests/path/to/test_file.py::test_function_name -v`
- Run tests with coverage: `python -m pytest --cov=src/file tests/`
- Run linter: `ruff check src/ tests/`
- Format code: `ruff format src/ tests/`


## Technical Guidelines


### Project Management
- **Python version**: Python 3.11+ is preferred
- **Project config**: `pyproject.toml` for configuration and dependency management
- **Environment**: Use virtual environment in `.venv` for dependency isolation
- **Package management**: Use `uv` for faster, more reliable dependency management with lock file
- **Dependencies**: Separate production and dev dependencies in `pyproject.toml`
- **Version management**: Use `setuptools_scm` for automatic versioning from Git tags
- **Linting**: `ruff` for style and error checking
- **Type checking**: Use VS Code with Pylance for static type checking
- **Project layout**: Organize code with `src/` layout

### Code Style Guidelines

- **Formatting**: Black-compatible formatting via `ruff format`
- **Imports**: Sort imports with `ruff` (stdlib, third-party, local)
- **Type hints**: Use native Python type hints (e.g., `list[str]` not `List[str]`)
- **Documentation**: Google-style docstrings for all modules, classes, functions
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Function length**: Keep functions short (< 30 lines) and single-purpose
- **PEP 8**: Follow PEP 8 style guide (enforced via `ruff`)

### Python Best Practices

- **File handling**: Prefer `pathlib.Path` over `os.path`
- **Debugging**: Use `logging` module instead of `print`
- **Error handling**: Use specific exceptions with context messages and proper logging
- **Data structures**: Use list/dict comprehensions for concise, readable code
- **Function arguments**: Avoid mutable default arguments
- **Data containers**: Leverage `dataclasses` to reduce boilerplate
- **Configuration**: Use environment variables (via `python-dotenv`) for configuration
- **Security**: Never store/log secret credentials, set command timeouts

### Development Patterns & Best Practices

- **Favor simplicity**: Choose the simplest solution that meets requirements
- **DRY principle**: Avoid code duplication; reuse existing functionality
- **Configuration management**: Use environment variables for different environments
- **Focused changes**: Only implement explicitly requested or fully understood changes
- **Preserve patterns**: Follow existing code patterns when fixing bugs
- **File size**: Keep files under 300 lines; refactor when exceeding this limit
- **Test coverage**: Write comprehensive unit and integration tests with `pytest`; include fixtures
- **Test structure**: Use table-driven tests with parameterization for similar test cases
- **Mocking**: Use unittest.mock for external dependencies; don't test implementation details
- **Modular design**: Create reusable, modular components
- **Logging**: Implement appropriate logging levels (debug, info, error)
- **Error handling**: Implement robust error handling for production reliability
- **Security best practices**: Follow input validation and data protection practices
- **Performance**: Optimize critical code sections when necessary
- **Dependency management**: Add libraries only when essential
  - When adding/updating dependencies, update `pyproject.toml` first
  - Regenerate the lock file with `uv pip compile --system pyproject.toml -o uv.lock`
  - Install the new dependencies with `uv pip sync --system uv.lock`

### Development Workflow

- **Version control**: Commit frequently with clear messages
- **Versioning**: Use Git tags for versioning (e.g., `git tag -a 1.2.3 -m "Release 1.2.3"`)
  - For releases, create and push a tag
  - For development, let `setuptools_scm` automatically determine versions
- **Impact assessment**: Evaluate how changes affect other codebase areas
- **Documentation**: Keep documentation up-to-date for complex logic and features
- **Dependencies**: When adding dependencies, always update the `uv.lock` file
- **CI/CD**: All changes should pass CI checks (tests, linting, etc.) before merging

## LOGBOOK.md Protocol

You must maintain a `LOGBOOK.md` file throughout your work. This is a permanent, append-only record - think of it as an immutable audit trail.

**Critical Rules:**
1. **NEVER delete or modify existing entries** - The logbook is append-only
2. **Always add new entries at the END of the file**
3. **Include timestamps for each entry**
4. **Document EVERYTHING** - mistakes, dead ends, solutions, realizations, todos

**Entry Format:**

```
[YYYY-MM-DD HH:MM] Entry Title

Context: What you were trying to do

Action: What you actually did

Result: What happened (including errors/failures)

```

**What to Log:**
- Initial understanding of the task
- Each approach attempted (successful or not)
- Errors encountered and their full messages
- Solutions discovered
- Workarounds implemented
- TODOs and future considerations
- Assumptions made
- External resources consulted
- Any "aha!" moments or pattern recognitions
- Dead ends (these are valuable!)

**Remember:** The logbook is a historical record. Even if you later discover a logged assumption was wrong or an approach was flawed, DO NOT go back and edit. Instead, add a new entry explaining the correction. The journey is as important as the destination.

Create LOGBOOK.md if it doesn't exist, and update it frequently as you work.


## Epic 2 Documentation
- @docs/epic2/epic2_logbook.md - Epic 2 implementation logbook for one-touch benchmark runner
- @docs/epic2/one_touch_analysis.md - Comprehensive analysis and implementation plan for Epic 2

## Project Documentation
- @.claude/cicd_testing_strategy.md - CI/CD testing strategy and guidelines
- @.claude/liteLLM/Batching Completion()  liteLLM.md - LiteLLM batching documentation
- @.claude/liteLLM/Reliability - Retries, Fallbacks  liteLLM.md - LiteLLM reliability features
- @.claude/liteLLM/Router - Load Balancing  liteLLM.md - LiteLLM router and load balancing
- @.claude/liteLLM/Streaming + Async  liteLLM.md - LiteLLM streaming and async features

## Project Overview

FLaME (Financial Language Understanding Evaluation) is a framework for evaluating Large Language Models (LLMs) on financial tasks. The repository contains multiple datasets for evaluating model performance on tasks like sentiment analysis, causal classification, summarization, and more in the financial domain.

## Key Commands

### Project Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# Or
.\.venv\Scripts\activate  # On Windows

# Install dependencies using uv (preferred)
uv pip install -r requirements.txt

# Install FLaME in development mode
uv pip install -e .

# Test the installation
uv run python -c "import flame; print(flame.__file__)"
```

### Running Inference

Run inference on a dataset using a specific model:

```bash
# Using uv
uv run python main.py --config configs/default.yaml --mode inference --dataset <dataset_name> --model "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
```

Example:
```bash
uv run python main.py --config configs/default.yaml --mode inference --dataset fomc --model "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
```

Override parameters from config file:
```bash
uv run python main.py --config configs/default.yaml --mode inference --dataset fomc --model "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct" --max_tokens 256 --temperature 0.1 --prompt_format zero_shot
```

### Running Evaluation

Run evaluation on inference results:

```bash
uv run python main.py --config configs/default.yaml --mode evaluate --dataset <dataset_name> --file_name <results_file_path>
```

Example:
```bash
uv run python main.py --config configs/default.yaml --mode evaluate --dataset fomc --file_name "results/fomc/fomc_together_ai_meta-llama_Llama-4-Scout-17B-16E-Instruct_13_05_2025.csv"
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_all_inference.py

# Run specific test with verbose output
uv run pytest -vv tests/test_all_inference.py

# Run specific test matching a pattern
uv run pytest -k "fomc"

# Run test in debug mode
uv run pytest --pdb
```

### Installing Packages with uv

Always use `uv` for package management:

```bash
# Add a new package
uv add packagename

# Add a package with specific version
uv add packagename==1.2.3

# Run Python scripts
uv run python script.py

# Generate requirements.txt
uv pip compile pyproject.toml -o requirements.txt
```

## Project Architecture

### Core Components

1. **Data Management**: 
   - The `data/` directory contains subfolders for each dataset
   - Each dataset folder contains a `huggify_*.py` script to upload data to HuggingFace

2. **Inference Pipeline**:
   - `main.py`: Entry point that parses args and runs inference or evaluation
   - `src/flame/code/inference.py`: Core orchestrator for inference tasks
   - `src/flame/code/evaluate.py`: Core orchestrator for evaluation tasks
   - `src/flame/code/<dataset>/<dataset>_inference.py`: Task-specific inference logic
   - `src/flame/code/<dataset>/<dataset>_evaluate.py`: Task-specific evaluation logic
   - `src/flame/code/prompts.py`: Zero-shot prompts used for inference

3. **Configuration System**:
   - `configs/default.yaml`: Default configuration settings
   - Command-line arguments can override config file values
   - Configuration flows through application via args object

4. **Testing Framework**:
   - Smoke tests for all inference and evaluation modules
   - Mock/stub system in `conftest.py` ensures tests run offline and fast

### Directory Structure

```
FLaME/
├── configs/                # Configuration files
├── data/                   # Task-specific datasets
├── src/flame/              # Main source code
│   ├── code/               # Task implementations
│   │   ├── <dataset>/      # Dataset-specific code
│   │   │   ├── <dataset>_inference.py
│   │   │   └── <dataset>_evaluate.py
│   │   ├── inference.py    # Main inference orchestrator
│   │   ├── evaluate.py     # Main evaluation orchestrator
│   │   └── prompts.py      # Prompt templates
│   ├── utils/              # Utility modules
│   └── config.py           # Project-wide configuration
├── results/                # Inference results 
├── evaluations/            # Evaluation results
├── tests/                  # Test suite
│   ├── conftest.py         # Test fixtures and mocks
│   ├── test_all_inference.py # Test all inference modules
│   └── test_all_evaluation.py # Test all evaluation modules
└── claude/                 # Testing guides and documentation
```

### Code Flow

1. **Inference pipeline**:
   - `main.py` parses args and loads config
   - `inference.py` maps dataset name to task-specific inference function
   - Task function loads dataset, creates prompts, calls the LLM API, and returns a DataFrame
   - Results are saved to CSV in `results/<dataset>/`

2. **Evaluation pipeline**:
   - `main.py` parses args and loads config
   - `evaluate.py` maps dataset name to task-specific evaluation function
   - Evaluation function loads inference results, computes metrics, and returns DataFrames
   - Results are saved to CSV in `evaluations/<dataset>/`

## API Integration

The project uses LiteLLM as an abstraction layer for calling different LLM providers. By default, use the Together.ai provider with the Llama-4-Scout-17B-16E-Instruct model:

```yaml
model: "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
```

## Testing Framework

FLaME has a comprehensive testing framework:

1. **Test Discovery**: 
   - Tests dynamically discover inference and evaluation modules

2. **Stubbed External Calls**: 
   - LiteLLM API calls are mocked/stubbed
   - No network connections or API keys needed for tests

3. **Test Isolation**:
   - Tests run in isolation with temporary directories
   - Fast and deterministic

4. **Debug Helpers**:
   - Verbose test outputs with `-vv`
   - Interactive debugging with `--pdb`

## Testing Guides

For detailed guidance on the testing framework, refer to these resources:

- [Inference Testing Guide](./.claude/inference_testing_guide.md): Details on testing inference modules
- [Evaluation Testing Guide](./.claude/evaluation_testing_guide.md): Details on testing evaluation modules
- [Testing Suite Notes](./.claude/testing_suite_notes.md): Design and roadmap for testing

## Common Patterns

1. **Adding a new dataset**:
   - Create a new folder in `data/<new_dataset>/`
   - Create a `huggify_<new_dataset>.py` script if uploading to HuggingFace
   - Add task-specific code in `src/flame/code/<new_dataset>/`
   - Implement `<new_dataset>_inference.py` and `<new_dataset>_evaluate.py`
   - Add prompt generation in `src/flame/code/prompts.py`
   - Update the task map in `src/flame/code/inference.py` and `src/flame/code/evaluate.py`

2. **Creating prompts**:
   - Define a function in `prompts.py` that takes input text and returns formatted prompt
   - Follow existing pattern like:
     ```python
     def task_prompt(input_text: str) -> str:
         return f"""Instructions for task.
                    Input: {input_text}"""
     ```

3. **Working with results**:
   - Results are saved as CSVs in `results/<dataset>/`
   - Follow naming convention: `<dataset>_<provider>_<model>_<date>.csv`
   - Evaluation metrics are saved in `evaluations/<dataset>/`

4. **Batch processing**:
   - Use `chunk_list` from `flame.utils.batch_utils` to create batches
   - Process with `process_batch_with_retry` for resilient API calls

## Important Notes

1. **API Keys**:
   - Store API keys in a `.env` file in the project root
   - Required keys include:
     ```
     HUGGINGFACEHUB_API_TOKEN=<your_token>
     TOGETHER_API_KEY=<your_key>
     ```

2. **Default Model**:
   - Always use the Together.ai provider with Llama-4-Scout-17B-16E-Instruct model
   ```
   together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct
   ```

3. **Package Management**:
   - Always use `uv` for package management and running Python scripts
   - Examples:
     ```bash
     # Install packages
     uv add packagename
     
     # Run Python scripts
     uv run python script.py
     ```

4. **Testing**:
   - Run tests before making changes to ensure everything works
   - Tests use mock/stub libraries to avoid network calls

5. **Result Paths**:
   - Inference results: `results/<dataset>/<dataset>_<provider>_<model>_<date>.csv`
   - Evaluation results: `evaluations/<dataset>/evaluation_<dataset>_<model>_<date>.csv`

## Tasks Not in Current Release

The following tasks are not included in the current release version and are deferred for future implementation:

1. **EconLogicQA** - Not used in the camera-ready version of the paper
   - Status: Deferred for future release
   - Inference module has deprecation notice
   - Tests should be marked to skip

2. **MMLU** - Not used in the camera-ready version of the paper
   - Status: Deferred for future release
   - MMLU (Massive Multitask Language Understanding) benchmark
   - Inference and evaluation modules have deprecation notices
   - Tests should be marked to skip

## Project Specific Memories
- Always use `uv` package manager. Do not run python without using uv. Always use `uv add` then `uv pip install` but never bare python without uv.
- **NEVER run inference yourself** - do not use `uv run python main.py` ever on your own, let the user run that command
- Only run dataset loading tests and other non-inference commands
- Wait for inference to complete rather than interrupting
- NEVER run `uv run python main.py` on your own!