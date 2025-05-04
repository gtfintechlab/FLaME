# FLaME Testing Suite – Design & Roadmap

## 1. Current Status (2025-05-04)
### 1.1 Inference tests
* **File discovery** – `tests/test_all_inference.py` walks `src/flame/code/**/_inference.py`, converts paths to module names and parametrises a single smoke-test over all modules (24 as of today).
* **Execution** – For each module the test locates the first callable named `*_inference` (convention) and runs it with a shared `dummy_args` fixture.
* **Fixtures & Stubs** – `tests/conftest.py` contains an autouse fixture that guarantees *zero* outbound calls:
  | External | Patch behaviour |
  |----------|-----------------|
  | `litellm.completion`, `litellm.batch_completion` | Return synthetic `_FakeCompletion` objects (OpenAI-like) |
  | `datasets.load_dataset` | In-memory `_DummyDataset` (two `_DummyRow`s) |
  | `time.sleep` | no-op |
  | `together.Together` | Stub client with `chat.completions.create → _FakeCompletion` |
  | `nltk.download` | no-op |
  | `flame.code.tokens.tokens` | returns `[]` to suppress deprecation |
  | `flame.config.RESULTS_DIR`, `LOG_DIR` | Redirected to a temp dir under `pytest` workspace |
  | MMLU – `flame.code.mmlu.mmlu_loader.MMLULoader` | Stub that returns a 2-row DataFrame and 1 few-shot example |
  | Prompt aliasing | `tatqa_prompt` → `tatqa_zeroshot_prompt` |

* **Dummy data extensions** – `_DummyRow._DEFAULTS` provides task-specific keys so modules like FinRED / ReFinD that expect nested fields don’t crash.
* **Results** – `pytest -q tests` → `23 passed, 1 skipped` (banking77 uses unusual shape – intentionally skipped by module). Total runtime ~6 s.

### 1.2 Adding new inference tasks
1. Drop new `*_inference.py` under `src/flame/code/...`.
2. If it needs extra dataset keys or exotic library, extend stubs in `tests/conftest.py`.
3. Run `pytest`. The discovery test will include it automatically.

## 2. Roadmap – Evaluation Tests
We now need fast, offline smoke-tests for each `*_evaluate.py` script (19 found).

### 2.1 Observation of evaluation modules
Early scan shows patterns:
* Most read **prediction results** from `RESULTS_DIR` (CSV) and **ground-truth** from `datasets.load_dataset` (or embedded columns) then compute metrics via pandas / sklearn / custom logic.
* Outputs are often metrics JSON / printed scores; a DataFrame return is optional.

### 2.2 Testing goals
* Verify each evaluation function executes without errors given minimal dummy inputs.
* Ensure metric keys exist and are numeric (sanity check) – precision, F1, etc.
* Keep tests *offline & deterministic* (reuse existing mocking philosophy).

### 2.3 Stub strategy
1. **Prediction CSV** – Create a tiny DataFrame (2 rows) matching expected schema and patch `pandas.read_csv` or use `monkeypatch.setattr(Path, "open", ...)` inside fixture to return it via `StringIO`.
2. **Ground-truth dataset** – Already handled by `datasets.load_dataset` stub; may need to enrich `_DummyRow` for label columns.
3. **File existence** – Many evaluators search `RESULTS_DIR / task / *.csv`. During test, we’ll pre-create an empty temp dir and place the synthetic CSV into it (path derived from evaluator’s logic). Fixture can compute the path and write DataFrame to disk inside pytest temp folder.
4. **Metric libraries** – If sklearn or other heavy libs appear and are unavailable, patch with lightweight dummy metric functions returning 0.0.

### 2.4 Shared fixtures (to add)
* `dummy_results_dir(tmp_path_factory)` – yields path, writes 2-row CSV for each task before evaluation runs.
* Extend current autouse fixture to override `pandas.read_csv` when evaluator constructs path dynamically.

### 2.5 Parameterized test skeleton
```python
EVAL_SRC = SRC_DIR  # reuse from inference tests

def _discover_eval_modules():
    return sorted(p.with_suffix("").relative_to(SRC_DIR.parent.parent).parts for p in EVAL_SRC.rglob("*_evaluate.py"))

@pytest.mark.parametrize("module_name", _discover_eval_modules())
def test_eval_module(module_name, dummy_args, dummy_results_dir):
    mod = importlib.import_module(module_name)
    fn = [v for v in mod.__dict__.values() if callable(v) and v.__name__.endswith("_evaluate")][0]
    metrics = fn(dummy_args)  # or no args depending on signature
    assert metrics  # not None / empty
```

### 2.6 Task-specific tweaks
* Some evaluation functions expect **model name** in args or parse filename strings; we can set `dummy_args.model` accordingly.
* Unique schemas (e.g., FNXL extraction+classification) – extend CSV builder for those columns.

### 2.7 Next steps
1. Browse each `*_evaluate.py` quickly to catalogue required columns & signature.
2. Build a mapping `{task: csv_columns}` to generate dummy DataFrame.
3. Implement `dummy_results_csv` fixture creating correct file per task.
4. Add `tests/test_all_evaluation.py` similar to inference counterpart.
5. Update stubs as discrepancies arise.

---
*Maintainer tip*: keep stubs minimal but expandable; prefer patching over monkeypatching functions to avoid import-order pitfalls.
