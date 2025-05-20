# Epic 1: Selectable Multi-Task Execution of Inference and Evaluation

## GitHub Issue Information

### Epic Issue #87: E-1: Selectable multi-task execution of inference and evaluation
- **State:** OPEN
- **Assignees:** glennmatlin, mika-okamoto, ShayarShah, yahyahas3an, huzaifahp7
- **Milestone:** Version 0.2
- **Description:** Epic covering basic ability to have YAML-selectable tasks / evals and conduct inference and evaluation

### Component Issues

#### Issue #98: Parse tasks list from YAML
- **State:** OPEN
- **Assignee:** glennmatlin
- **Milestone:** Version 0.2
- **Labels:** type: feature, type: task
- **Description:** Implements config key `tasks:` to whitelist benchmarks.

#### Issue #99: Select multiple tasks for inference||evaluation
- **State:** OPEN
- **Assignee:** glennmatlin
- **Milestone:** Version 0.2
- **Labels:** area: cli, area: core, type: feature
- **Description:** Overrides YAML list at runtime.

#### Issue #100: Validate task names
- **State:** OPEN
- **Assignee:** glennmatlin
- **Labels:** area: cli, type: task
- **Description:** Block invalid names early (`flame list-tasks`).

#### Issue #101: Basic execution test suite
- **State:** OPEN
- **Assignee:** glennmatlin
- **Labels:** type: test
- **Description:** Unit tests
- **Comment:** Started first run at testing in PR #153 "First run at testing for all the inference and evaluation code" (now closed)

#### Issue #102: Documentation for selective-run
- **State:** OPEN
- **Assignee:** glennmatlin
- **Milestone:** Version 0.2
- **Labels:** type: doc
- **Description:** Add section to README with examples.

#### Issue #103: Wrapper script for "flame run all"
- **State:** OPEN
- **Assignee:** glennmatlin
- **Milestone:** Version 0.2
- **Labels:** area: cli, area: core, type: task
- **Description:** Entry-point that orchestrates full pipeline.

#### Issue #105: CI pipeline smoke test
- **State:** OPEN
- **Milestone:** Version 0.2
- **Labels:** type: ci
- **Description:** GitHub Action that runs tiny dataset to verify the basic smoke tests, cli usage, etc.

### Related Code and Branches

- **Current Branch:** `refactor/epic1/multitask`
- **Recent Commits:**
  - `fb38e2f` merge in main
  - `fa6f0da` epic1/cleanup (#155)
  - `2447a9e` Add VSCode settings for Python type checking and pytest configuration
  - `3893903` First run at testing for all the inference and evaluation code
  - `d5fdb5a` uv updates

## Overview

This document outlines the implementation plan for Epic 1 (issue #87): "Selectable multi-task execution of inference and evaluation". The goal is to enable users to specify multiple tasks for running inference and evaluation either through YAML configuration or command-line arguments, making the FLaME framework more flexible and powerful.

## Current State Assessment

The codebase currently:
- Requires separate runs for each task
- Has no built-in validation for task names
- Lacks a unified way to execute multiple tasks
- Has initial testing infrastructure but needs more test coverage

The current branch `refactor/epic1/multitask` has begun implementing these features, and PR #153 has added initial testing infrastructure.

## Goals and Success Criteria

1. Users can specify multiple tasks in YAML configuration
2. Users can override YAML task list via command-line
3. Invalid task names are rejected with helpful error messages
4. A single command can run multiple tasks in sequence
5. Comprehensive test coverage ensures functionality
6. CI pipeline verifies basic functionality
7. Documentation clearly explains the new features

## Implementation Plan and Dependencies

### Task Dependency Graph

```
                        [#98: Parse tasks list from YAML]
                           /               \
                          /                 \
         [#100: Validate task names]   [#99: Select multiple tasks]
                         \                  /
                          \                /
                      [#103: Wrapper script for "flame run all"]
                           /               \
                          /                 \
                [#101: Test suite]    [#102: Documentation]
                          \                /
                           \              /
                        [#105: CI pipeline]
```

### Phase 1: Core Functionality

#### Task 1: Parse tasks list from YAML (Issue #98)
- **Dependencies:** None
- **Steps:**
  1. Modify `configs/default.yaml` to add a `tasks` list field:
     ```yaml
     # Core settings
     dataset: fomc  # Default single task (for backward compatibility)
     tasks:         # New multi-task list
       - fomc
       - fnxl
       - numclaim
     mode: inference  # inference or evaluate
     ```
  2. Update the config loading in `main.py` to parse this list
  3. Add logic to use this list when no single dataset is specified
  4. Update type hints and configuration validation
- **Acceptance Criteria:**
  - Config file with `tasks` list loads correctly
  - If `tasks` is present, it overrides single `dataset` parameter
  - Empty task list produces appropriate error message
- **Potential Pitfalls:**
  - Backward compatibility issues with existing configs
  - Handling edge cases like duplicate task names

#### Task 2: Validate task names (Issue #100)
- **Dependencies:** Task registry needs to be available
- **Steps:**
  1. Create a task registry module that centralizes valid task names:
     ```python
     # src/flame/task_registry.py
     from flame.code.inference import task_inference_map
     from flame.code.evaluate import task_evaluate_map

     def get_available_inference_tasks():
         """Return list of valid inference task names."""
         return list(task_inference_map.keys())

     def get_available_evaluation_tasks():
         """Return list of valid evaluation task names."""
         return list(task_evaluate_map.keys())

     def is_valid_task(task_name, mode="inference"):
         """Check if a task name is valid for the given mode."""
         if mode == "inference":
             return task_name in get_available_inference_tasks()
         elif mode == "evaluate":
             return task_name in get_available_evaluation_tasks()
         return False

     def validate_tasks(tasks, mode="inference"):
         """Validate a list of task names for the given mode.
         
         Returns:
             tuple: (is_valid, invalid_tasks)
         """
         if mode == "inference":
             valid_tasks = set(get_available_inference_tasks())
         else:
             valid_tasks = set(get_available_evaluation_tasks())
             
         invalid_tasks = [t for t in tasks if t not in valid_tasks]
         return (len(invalid_tasks) == 0, invalid_tasks)
     ```
  2. Implement CLI `list-tasks` command in `main.py`
     ```python
     if args.subcommand == "list-tasks":
         print("Available inference tasks:")
         for task in sorted(get_available_inference_tasks()):
             print(f"  - {task}")
         print("\nAvailable evaluation tasks:")
         for task in sorted(get_available_evaluation_tasks()):
             print(f"  - {task}")
         return
     ```
  3. Add validation early in the main execution flow
- **Acceptance Criteria:**
  - Command `python main.py list-tasks` shows all available tasks
  - Invalid task names result in clear error message with suggestions
  - Users can see all available tasks for both inference and evaluation modes
- **Potential Pitfalls:**
  - Tasks might have different availability between inference and evaluation
  - Need to handle case sensitivity and similar task names

#### Task 3: Select multiple tasks at runtime (Issue #99)
- **Dependencies:** Tasks #1 and #2
- **Steps:**
  1. Modify argument parser in `main.py` to accept multiple tasks:
     ```python
     parser.add_argument(
         "--tasks", 
         type=str, 
         nargs="+", 
         help="List of task names to run (overrides config)"
     )
     ```
  2. Update argument processing to handle both single dataset and tasks list
  3. Prioritize command-line arguments over config file values
- **Acceptance Criteria:**
  - Command-line specified tasks override config file
  - Both `--dataset single_task` and `--tasks task1 task2 task3` work
  - Multiple tasks specified at runtime are validated
- **Potential Pitfalls:**
  - Conflicts between dataset and tasks parameters
  - Preserving backward compatibility

### Phase 2: Integration and Execution

#### Task 4: Create wrapper script for "flame run all" (Issue #103)
- **Dependencies:** Tasks #1, #2, and #3
- **Steps:**
  1. Create a multi-task execution orchestrator:
     ```python
     # src/flame/run_multi.py
     
     from flame.code.inference import main as inference
     from flame.code.evaluate import main as evaluate
     from flame.task_registry import validate_tasks
     import copy
     
     def run_multi_task(args):
         """Run multiple tasks sequentially.
         
         Args:
             args: Command line arguments with tasks list
         
         Returns:
             dict: Results summary for each task
         """
         results = {}
         
         # Validate all tasks first
         mode = args.mode
         is_valid, invalid_tasks = validate_tasks(args.tasks, mode=mode)
         if not is_valid:
             raise ValueError(f"Invalid tasks for {mode}: {', '.join(invalid_tasks)}")
         
         # Run each task with its own args copy
         for task in args.tasks:
             task_args = copy.deepcopy(args)
             task_args.dataset = task
             
             print(f"\n{'='*40}\nRunning {mode} for task: {task}\n{'='*40}\n")
             
             try:
                 if mode == "inference":
                     result = inference(task_args)
                 elif mode == "evaluate":
                     result = evaluate(task_args)
                 results[task] = {"status": "success", "result": result}
             except Exception as e:
                 results[task] = {"status": "failed", "error": str(e)}
                 print(f"Error running {mode} for {task}: {str(e)}")
         
         # Print summary
         print(f"\n{'='*40}\nExecution Summary\n{'='*40}")
         for task, result in results.items():
             status = "✅" if result["status"] == "success" else "❌"
             print(f"{status} {task}")
         
         return results
     ```
  2. Integrate with `main.py` to handle multi-task execution
  3. Implement progress tracking and error handling
- **Acceptance Criteria:**
  - Single command can run multiple tasks in sequence
  - Failures in one task don't stop execution of other tasks
  - Clear summary of results is shown
- **Potential Pitfalls:**
  - Handling task-specific parameters
  - Resource management for running multiple tasks
  - Error propagation and reporting

### Phase 3: Testing and Documentation

#### Task 5: Create basic execution test suite (Issue #101)
- **Dependencies:** Tasks #1, #2, #3, and #4
- **Steps:**
  1. Create unit tests for task registry:
     ```python
     # tests/test_task_registry.py
     import pytest
     from flame.task_registry import (
         get_available_inference_tasks,
         get_available_evaluation_tasks,
         is_valid_task,
         validate_tasks
     )
     
     def test_get_available_tasks():
         """Test that available tasks are returned."""
         inference_tasks = get_available_inference_tasks()
         evaluation_tasks = get_available_evaluation_tasks()
         
         assert isinstance(inference_tasks, list)
         assert isinstance(evaluation_tasks, list)
         assert len(inference_tasks) > 0
         assert len(evaluation_tasks) > 0
     
     def test_is_valid_task():
         """Test task validation."""
         # Get a valid task from the registry
         valid_task = get_available_inference_tasks()[0]
         
         assert is_valid_task(valid_task, mode="inference")
         assert not is_valid_task("not_a_real_task", mode="inference")
     
     def test_validate_tasks():
         """Test validation of multiple tasks."""
         valid_tasks = get_available_inference_tasks()[:2]
         invalid_tasks = ["not_a_task", "another_invalid"]
         mixed_tasks = valid_tasks + invalid_tasks
         
         is_valid, invalid = validate_tasks(valid_tasks, mode="inference")
         assert is_valid
         assert len(invalid) == 0
         
         is_valid, invalid = validate_tasks(invalid_tasks, mode="inference")
         assert not is_valid
         assert set(invalid) == set(invalid_tasks)
         
         is_valid, invalid = validate_tasks(mixed_tasks, mode="inference")
         assert not is_valid
         assert set(invalid) == set(invalid_tasks)
     ```
  2. Create tests for YAML parsing:
     ```python
     # tests/test_yaml_parsing.py
     import pytest
     import yaml
     from pathlib import Path
     import tempfile
     
     # Import the function that loads config
     from main import parse_arguments
     
     def test_parse_single_dataset():
         """Test parsing config with single dataset."""
         with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
             config = {
                 "dataset": "fomc",
                 "mode": "inference"
             }
             yaml.dump(config, tmp)
             tmp.flush()
             
             args = parse_arguments(["--config", tmp.name])
             assert args.dataset == "fomc"
             assert not hasattr(args, "tasks") or args.tasks is None
     
     def test_parse_tasks_list():
         """Test parsing config with tasks list."""
         with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
             config = {
                 "tasks": ["fomc", "numclaim"],
                 "mode": "inference"
             }
             yaml.dump(config, tmp)
             tmp.flush()
             
             args = parse_arguments(["--config", tmp.name])
             assert hasattr(args, "tasks")
             assert args.tasks == ["fomc", "numclaim"]
     
     def test_cli_override():
         """Test CLI args override config file."""
         with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
             config = {
                 "tasks": ["fomc", "numclaim"],
                 "mode": "inference"
             }
             yaml.dump(config, tmp)
             tmp.flush()
             
             # Override with --tasks
             args = parse_arguments(["--config", tmp.name, "--tasks", "fnxl", "finer"])
             assert args.tasks == ["fnxl", "finer"]
             
             # Override with --dataset (single task)
             args = parse_arguments(["--config", tmp.name, "--dataset", "fnxl"])
             assert args.dataset == "fnxl"
             assert not hasattr(args, "tasks") or args.tasks is None
     ```
  3. Create tests for multi-task execution:
     ```python
     # tests/test_multi_task.py
     import pytest
     from types import SimpleNamespace
     
     # Mock the inference and evaluate functions
     def mock_inference(args):
         return {"status": "success", "task": args.dataset}
     
     def mock_evaluate(args):
         return {"status": "success", "task": args.dataset}
     
     # Patch the actual functions for testing
     @pytest.fixture
     def patch_execution(monkeypatch):
         import flame.code.inference
         import flame.code.evaluate
         
         monkeypatch.setattr(flame.code.inference, "main", mock_inference)
         monkeypatch.setattr(flame.code.evaluate, "main", mock_evaluate)
         
         # Also patch the task registry to use mock tasks
         import flame.task_registry
         monkeypatch.setattr(
             flame.task_registry, 
             "get_available_inference_tasks", 
             lambda: ["mock_task1", "mock_task2"]
         )
         monkeypatch.setattr(
             flame.task_registry, 
             "get_available_evaluation_tasks", 
             lambda: ["mock_task1", "mock_task2"]
         )
     
     def test_run_multi_task(patch_execution):
         """Test running multiple tasks."""
         from flame.run_multi import run_multi_task
         
         # Create test args
         args = SimpleNamespace(
             tasks=["mock_task1", "mock_task2"],
             mode="inference"
         )
         
         # Run the multi-task function
         results = run_multi_task(args)
         
         # Verify results
         assert len(results) == 2
         assert "mock_task1" in results
         assert "mock_task2" in results
         assert results["mock_task1"]["status"] == "success"
         assert results["mock_task2"]["status"] == "success"
     
     def test_run_multi_task_with_failure(patch_execution, monkeypatch):
         """Test running multiple tasks with one failing."""
         from flame.run_multi import run_multi_task
         
         # Make the first task fail
         def mock_inference_with_failure(args):
             if args.dataset == "mock_task1":
                 raise ValueError("Test error")
             return {"status": "success", "task": args.dataset}
         
         import flame.code.inference
         monkeypatch.setattr(flame.code.inference, "main", mock_inference_with_failure)
         
         # Create test args
         args = SimpleNamespace(
             tasks=["mock_task1", "mock_task2"],
             mode="inference"
         )
         
         # Run the multi-task function
         results = run_multi_task(args)
         
         # Verify results
         assert len(results) == 2
         assert results["mock_task1"]["status"] == "failed"
         assert "error" in results["mock_task1"]
         assert results["mock_task2"]["status"] == "success"
     ```
- **Acceptance Criteria:**
  - All tests pass and cover core functionality
  - Edge cases are tested (invalid tasks, error handling)
  - Integration tests verify end-to-end functionality
- **Potential Pitfalls:**
  - Mocking complex dependencies
  - Testing error handling across multiple tasks
  - Test environment differences

#### Task 6: Create documentation (Issue #102)
- **Dependencies:** All previous tasks
- **Steps:**
  1. Update README.md with examples:
     ```markdown
     ## Multi-Task Execution
     
     FLaME supports running multiple tasks in a single command:
     
     ### Using YAML Configuration
     
     Create a config file with a `tasks` list:
     
     ```yaml
     # multi_task_config.yaml
     tasks:
       - fomc
       - numclaim
       - fnxl
     mode: inference
     model: "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
     max_tokens: 128
     temperature: 0.0
     ```
     
     Run with:
     
     ```bash
     uv run python main.py --config multi_task_config.yaml
     ```
     
     ### Using Command-Line Arguments
     
     You can specify tasks directly on the command line:
     
     ```bash
     uv run python main.py --config default.yaml --tasks fomc numclaim fnxl --mode inference
     ```
     
     ### List Available Tasks
     
     View all available tasks with:
     
     ```bash
     uv run python main.py list-tasks
     ```
     ```
  2. Create a detailed guide in `/docs/multi_task_guide.md`
- **Acceptance Criteria:**
  - README includes clear examples of multi-task execution
  - Documentation explains all new features
  - Examples cover common use cases
- **Potential Pitfalls:**
  - Ensuring documentation stays up-to-date with implementation
  - Covering all edge cases and options

#### Task 7: CI pipeline smoke test (Issue #105)
- **Dependencies:** All previous tasks
- **Steps:**
  1. Create GitHub Actions workflow:
     ```yaml
     # .github/workflows/smoke_test.yml
     name: Multi-Task Smoke Test
     
     on:
       push:
         branches: [ main, refactor/epic1/multitask ]
       pull_request:
         branches: [ main ]
     
     jobs:
       smoke-test:
         runs-on: ubuntu-latest
         
         steps:
         - uses: actions/checkout@v2
         
         - name: Set up Python
           uses: actions/setup-python@v2
           with:
             python-version: '3.11'
             
         - name: Install uv
           run: |
             curl -LsSf https://astral.sh/uv/install.sh | sh
             
         - name: Install dependencies
           run: |
             uv pip install -e .
             
         - name: Run tests
           run: |
             uv run pytest tests/
             
         - name: List available tasks
           run: |
             uv run python main.py list-tasks
             
         - name: Run multi-task inference with tiny sample
           run: |
             echo "tasks: [fomc, numclaim]
             mode: inference
             model: together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct
             max_tokens: 16
             batch_size: 2
             sample_size: 2
             " > smoke_test_config.yaml
             uv run python main.py --config smoke_test_config.yaml
     ```
- **Acceptance Criteria:**
  - CI pipeline runs on PRs and main branch
  - Smoke test verifies basic functionality
  - Test failures are clearly reported
- **Potential Pitfalls:**
  - API access during CI runs (may need mocking)
  - Test environment differences
  - Execution time in CI environment

## Potential Challenges and Mitigations

1. **Backward Compatibility**
   - **Challenge:** Maintaining compatibility with existing scripts and configs
   - **Mitigation:** Ensure single-task mode still works, prioritize parameters correctly

2. **Task Parameter Conflicts**
   - **Challenge:** Handling task-specific parameters that might conflict
   - **Mitigation:** Create isolated parameter contexts for each task execution

3. **Error Handling**
   - **Challenge:** Graceful error handling across multiple tasks
   - **Mitigation:** Implement robust error handling that allows continuation

4. **Performance**
   - **Challenge:** Running multiple large models sequentially could be resource-intensive
   - **Mitigation:** Add resource management and optional delays between tasks

5. **Testing Complexity**
   - **Challenge:** Testing all combinations of tasks could be overwhelming
   - **Mitigation:** Focus on parametrized tests and edge cases

## Timeline and Milestones

### Week 1: Core Functionality
- Complete Tasks #1 and #2
- Initial implementation of Task #3

### Week 2: Integration
- Complete Task #3
- Implement Task #4
- Begin Task #5 (testing)

### Week 3: Testing and Documentation
- Complete Task #5
- Implement Task #6
- Begin Task #7

### Week 4: Finalization
- Complete Task #7
- Final integration testing
- Address feedback and refinements

## Questions and Clarifications

1. Should we prioritize certain tasks for the initial implementation?
2. Are there specific task combinations that are particularly important to test?
3. Do we need to handle task-specific parameters differently for multi-task execution?
4. Should we implement a parallel execution option or keep it sequential?
5. How should we handle and report errors when running multiple tasks?

## Implementation Checklist

- [ ] Task #1: Parse tasks list from YAML (Issue #98)
- [ ] Task #2: Validate task names (Issue #100)
- [ ] Task #3: Select multiple tasks at runtime (Issue #99)
- [ ] Task #4: Create wrapper script for "flame run all" (Issue #103)
- [ ] Task #5: Create basic execution test suite (Issue #101)
- [ ] Task #6: Create documentation (Issue #102)
- [ ] Task #7: CI pipeline smoke test (Issue #105)