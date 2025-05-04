# FLaME – Evaluation Testing Guide

_Last updated: 2025-05-04_

This guide explains **how the offline evaluation smoke-test suite works**, how we keep every module isolated from external services, and how to extend the suite when new `*_evaluate.py` scripts are added.

---

## 1. File layout

| Path | Purpose |
|------|---------|
| `tests/test_all_evaluation.py` | Parametrised pytest that discovers every evaluation module and executes it with dummy data & heavily-stubbed environment. |
| `tests/conftest.py` | Global fixtures (shared with inference tests) + new patches for evaluation-specific quirks. |
| `agent/testing_suite_notes.md` | High-level roadmap (kept for history). |
| `agent/evaluation_testing_guide.md` | **THIS FILE** – implementation reference. |

---

## 2. Discovery logic

```python
SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "flame" / "code"
EVAL_MODULES = sorted(
    p.with_suffix("").relative_to(SRC_DIR.parent.parent).as_posix().replace("/", ".")
    for p in SRC_DIR.rglob("*_evaluate.py")
)
```

The test is parametrised over `EVAL_MODULES`. For each module we lazily import it _after_ all monkey-patches are installed to guarantee our stubs win the import-order race.

---

## 3. Dummy dataframe factory – `_make_dummy_df()`

Located at the top of the test file, this helper creates **one row** that satisfies **every column ever accessed** across the 23 evaluation scripts.

Highlights:

* Generic columns – `actual_labels`, `llm_responses`, `response`, etc.
* Task-specific extras – causal detection uses `complete_responses`, FiQA uses `actual_answers`, NumClaim wants numerical labels, FinEntity wants nested JSON strings, etc.
* Subjective-QA: loops over six label dimensions and registers both `_actual_label` and `_response` for each.

If a module asks for an unexpected column the DataFrame subclass `_DummyEvalDF` automatically creates it on-the-fly returning `None` values, preventing `KeyError`s.

---

## 4. Core patches & stubs

All live inside the test **before** the module import.

| Target | Why | Stub Behaviour |
|--------|-----|----------------|
| `pandas.read_csv` | Avoid filesystem | Always returns the shared dummy DF. |
| `sklearn.metrics.*` | Shape mismatch & heavy deps | All metric fns return `0.0` (or tuple of zeroes). |
| `builtins.eval` | Causal-detection parses model output with `eval` | Returns a fake `_FakeCompletion` object. |
| `pathlib.Path.exists` | NumClaim verifies annotation file on disk | Always `True`. |
| `json.loads` & `ast.literal_eval` | Some evaluators hand malformed strings | Safe wrappers returning `{}` (for JSON) or `[]` (for literals) on error. |
| `EVALUATION_DIR` | Prevent test outputs in main evaluation folder | Monkeypatched to a temporary `evaluation` dir under `pytest` workspace |
| `evaluate.load` (🤗 Hub metric) | ECTSum depends on BERTScore | Returns lightweight object whose `compute()` yields lists of zeroes. |
| Heavy libraries (`transformers`, `PIL`, etc.) | Optional visual models | Insert empty `SimpleNamespace` stubs. |

These patches are _local_ to each test invocation via `pytest.monkeypatch`.

---

## 5. Execution strategy

```
mod = importlib.import_module(module_name)
fn  = next(v for v in mod.__dict__.values() if callable(v) and v.__name__.endswith("_evaluate"))

sig = inspect.signature(fn)
if len(sig.parameters) >= 2:
    fn("dummy.csv", dummy_args)
elif len(sig.parameters) == 1:
    fn(dummy_args)
else:
    fn()
```

We call the function according to its signature pattern. Any exception bubbles up and the parametrised test fails with a helpful message.

Outcome after the latest run:

```bash
pytest -q tests/test_all_evaluation.py
.s..........s..s..........  [100%]
23 passed, 3 skipped, 26 warnings in 1.7s
```

No external network, GPU, or big models are touched.

---

## 6. Adding a new evaluation module

1. Drop your `*_evaluate.py` under `src/flame/code/<task>/`.
2. Run `pytest`. If it fails, inspect the traceback:
   * **Missing column** → add to `_make_dummy_df` or handle in `_DummyEvalDF` if appropriate.
   * **Un-stubbed library** → extend the patch table.
3. Keep the evaluation function signature similar (`evaluation(file_name, args)` preferred) for automatic discovery.

The guiding principle is: _tests must be offline, finish in under 10 s, and never charge API tokens._

---

## 7. Relationship with inference tests

Both suites share the same philosophy and many fixtures in `conftest.py`. Inference tests focus on **prompt generation / LLM call** pipelines, whereas evaluation tests validate **metric computation pipelines**. Because evaluation modules sometimes reuse completion utilities (e.g., for re-asking the model), we retain the earlier LLM stubs – they now serve both suites.

If you introduce a dependency that affects **both** paths, update `conftest.py` _once_.

---

## 8. Debug tips

* Add `-vv` to `pytest` to see the module name before failure.
* Touch a module name pattern with `-k headlines` to rerun just that test.
* Use `pytest --pdb` to drop into the failing evaluation and inspect the dummy DataFrame.

---

Happy testing! 🧪
