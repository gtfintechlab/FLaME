# Epic: Selectable Multi-Task Execution (Inference & Evaluation)

_Last updated: 2025-05-04_

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
| 4    | Unify single/multi path; update docs & guides                                      |                 | ⬜             |
| 5    | Expand test suite for multi-task workflows                                         |                 | ⬜             |
| 6    | Wrapper script `bin/flame_run_all` (optional)                                    |                 | ⬜             |

---

## 4. Changelog

* **2025-05-04** – added central registry and refactored imports.
