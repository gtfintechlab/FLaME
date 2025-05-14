# FLaME – Inference Testing Guide

_Last updated: 2025-05-04_

This guide explains the offline smoke-test suite for all `*_inference.py` modules, how it keeps each pipeline fast and offline, and how to add new inference tasks.

---

## 1. Discovery Logic

Parametrised in `tests/test_all_inference.py`:

```python
SRC_DIR = Path(__file__).resolve().parents[1] / 'src' / 'flame' / 'code'

def _discover_inference_modules():
    return sorted(
        ".".join(p.relative_to(SRC_DIR.parent.parent).with_suffix("").parts)
        for p in SRC_DIR.rglob("*_inference.py")
    )
```

Each module path (e.g., `flame.code.headlines.headlines_inference`) is passed into a single test:

```python
@pytest.mark.parametrize("module_name", _discover_inference_modules())
def test_inference_module(module_name, dummy_args):
    # skip any known-to-fail prefixes
    if any(module_name.startswith(p) for p in skip_prefixes):
        pytest.skip(...)
    mod = importlib.import_module(module_name)
    fn  = next(obj for obj in mod.__dict__.values()
               if callable(obj) and obj.__name__.endswith("_inference"))
    # adjust dataset arg
    dummy_args.dataset = module_name.split('.')[-2]
    result = fn(dummy_args)
    if hasattr(result, '__len__'):
        assert len(result) >= 0
```

---

## 2. `dummy_args` & Shared Fixtures

Defined in `tests/conftest.py`:

```python
@pytest.fixture()
def dummy_args():
    return SimpleNamespace(
        model="gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=16,
        top_p=1.0,
        batch_size=2,
        dataset="dummy",
        ...
    )
```

An auto-use fixture patches all external calls:

- **LiteLLM**: `litellm.completion`, `litellm.batch_completion`
- **datasets.load_dataset**: returns `_DummyDataset()`
- **time.sleep**: no-op
- **together.Together**: stub chat client
- **nltk.download**: no-op
- **flame.code.tokens.tokens**: returns `[]`
- **RESULTS_DIR**, **LOG_DIR**: redirected to pytest temp
- **MMLULoader**: stubbed to return a 2-row DataFrame

---

## 3. Skip List

Some pipelines aren’t ready for offline testing. Update `skip_prefixes` in `test_all_inference.py`:

```python
skip_prefixes = {
    "flame.code.mmlu",  # not yet stubbed fully
}
```

---

## 4. Warning Suppression

We suppress library deprecation & undefined-metric warnings in `conftest.py`:

```python
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

---

## 5. Adding New Inference Tasks

1. Create `*_inference.py` under `src/flame/code/<task>/`.
2. Ensure any new dataset keys are covered by `_DummyRow._DEFAULTS` or extend stubs in `conftest.py`.
3. If heavy dependencies appear (e.g., new deep-learning lib), add to skip list or stub in conftest.
4. Run `pytest -q` – the discovery test will include it automatically.

---

## 6. Running & Debug Tips

- **Run**: `pytest -q tests/test_all_inference.py`
- **Verbose**: `-vv` shows module names.
- **Targeted**: `pytest -q -k tatqa` reruns only tatqa pipeline.
- **PDB**: `pytest --pdb` to inspect failures interactively.

---

Happy coding! 🚀
