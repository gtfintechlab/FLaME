# Epic: Selectable Multi-Task Execution (Inference & Evaluation)

_Last updated: 2025-05-05_

This document tracks the high-level plan and implementation status for enabling FLaME to run **one or many tasks** with a single unified interface.

---

## 1. Goals

* Allow CLI & YAML to accept **one or more** tasks instead of the single `dataset` arg.
* Remove duplicated task maps; use a central registry.
* Sequential execution is sufficient for v1 (parallelism can come later).
* Expand smoke-test suite to cover multi-task invocations.

---

## 2. Architecture Changes

| Layer               | Before                                                 | After                                                                                        |
| ------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| **main.py**   | `--dataset` / `dataset:` (single)                  | `--tasks a b c` **or** YAML `tasks: [a,b]` (list). Single-task = list of length 1. |
| **Task maps** | Hard-coded dicts in `inference.py` & `evaluate.py` | `flame.task_registry` – single source-of-truth.                                           |
| **Runner**    | `if mode==inference: inference(args)`                | `run_tasks(tasks, mode, args)` loops over list.                                            |
| **Tests**     | `test_all_inference.py` parametrises each module     | New `test_multi_inference.py`, `test_yaml_parsing.py`.                                   |

---

## 3. Milestones & Status

| Step | Description                                                                        | PR              | Status         |
| ---- | ---------------------------------------------------------------------------------- | --------------- | -------------- |
| 1    | Create `task_registry.py`; refactor `inference.py` & `evaluate.py` to use it | _this commit_ | **Done** |
| 2    | Add `--tasks` + YAML parsing in `main.py`; deprecate `--dataset`             | _this commit_ | **Done** |
| 3    | Implement `run_tasks` function (sequential)                                      | _this commit_ | **Done** |
| 4    | Unify single/multi path; update docs & guides                                      | _this commit_ | **Done** |
| 5    | Expand test suite for multi-task workflows                                         | _this commit_ | **Done** |
| 6    | Fix test hanging issues in evaluation modules                                    | _this commit_ | **Done** |

---

## 4. Changelog

* **2025-05-04** – Added central registry and refactored imports.
* **2025-05-05** – Improved YAML config merging with CLI args; fixed test hanging issues; expanded test suite.

## 5. Implementation Details

### 5.1 Argument Parsing

The `parse_arguments` function in `main.py` now:
1. Parses CLI arguments first
2. Loads YAML config if specified
3. Merges YAML values where CLI args are not provided
4. Applies sensible defaults for any remaining unspecified arguments

This gives proper precedence order: CLI > YAML > defaults.

### 5.2 Tests

Multiple test files now cover the multi-task functionality:
- `test_multi_inference.py` - Tests running multiple inference tasks sequentially
- `test_yaml_parsing.py` - Tests CLI/YAML config merging and defaults
- `test_run_tasks_errors.py` - Tests error aggregation and reporting

### 5.3 Development Setup

The README now includes up-to-date instructions for development setup using `uv`:
- Creating and managing virtual environments
- Including the local `flame` package as a dependency
- Windows-specific absolute path considerations

### 5.4 Test Stability Improvements

- Fixed hanging tests in evaluation modules by removing external API calls
- Modified `test_yaml_parsing.py` to create temporary YAML files for testing
