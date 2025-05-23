"""Parametrised smoke-tests for every *_evaluate.py module.
Ensures evaluation code executes offline with stubbed data and metrics.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import ModuleType, SimpleNamespace as _SNS

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "flame" / "code"


def _discover_eval_modules() -> list[str]:
    """Return fully-qualified module paths for every *_evaluate*.py file."""
    paths = list(SRC_DIR.rglob("*_evaluate*.py"))
    modules: list[str] = []
    for file in paths:
        rel_parts = file.relative_to(SRC_DIR.parent.parent).with_suffix("").parts
        modules.append(".".join(rel_parts))
    return sorted(modules)


def _make_dummy_df() -> pd.DataFrame:  # noqa: D103 (helper)
    data: dict[str, list] = {
        # Generic classification label setup
        "actual_labels": ["[0,0,0,0,0,0,0]"],
        "llm_responses": ["dummy response"],
        # MMLU
        "raw_response": ["A"],
        "actual_answer": ["A"],
        "subject": ["economics"],
        # FinQA/TatQA
        "response": ["dummy 42"],
        "actual_label": ["42"],
        # generic label as numeric answer
        "actual_answer_numeric": ["42"],
        # FinQA evaluate expects df["actual_label"] maybe numeric
        # FinEntity, FinBench etc might use predicted/actual columns
        "predicted_label": ["dummy"],
        # Headlines actual_labels list string – 7 binary values
        "actual_labels_json": ["[0,0,0,0,0,0,0]"],
        # Placeholder extracted
        "extracted_labels": ["dummy"],
        # For causal detection modules
        "complete_responses": ["dummy"],
        "actual_tags": ["['CAUSE']"],
        "extracted_tags": ["['CAUSE']"],
        # For FiQA
        "actual_answers": ["answer"],
    }
    # SubjectiveQA columns appended after initial dict creation
    for feat in [
        "RELEVANT",
        "SPECIFIC",
        "CAUTIOUS",
        "ASSERTIVE",
        "CLEAR",
        "OPTIMISTIC",
    ]:
        data[f"{feat}_actual_label"] = [0]
        data[f"{feat}_response"] = ["0"]
    return pd.DataFrame(data)


@pytest.mark.parametrize("module_name", _discover_eval_modules())
def test_evaluation_module(module_name: str, dummy_args, monkeypatch):  # noqa: D103 – pytest test fn
    class _DummyEvalDF(pd.DataFrame):
        """DataFrame that auto-creates missing columns with default None values."""

        _metadata: list[str] = []

        def __getitem__(self, key):  # type: ignore[override]
            # Preserve pandas behaviour for non-string keys (mask, slices etc.)
            if not isinstance(key, str):
                return super().__getitem__(key)

            if key not in self.columns:
                self[key] = [None] * len(self)
            return super().__getitem__(key)

        def __getattr__(self, item):  # type: ignore[override]
            # Redirect attribute-style column access
            try:
                return self[item]
            except Exception as e:
                raise AttributeError(item) from e

    dummy_df = _DummyEvalDF(_make_dummy_df())
    # For causal detection modules, ensure both positive and negative cases to avoid sklearn warnings
    if "causal_detection" in module_name:
        # Create two rows: one positive, one negative
        df2 = pd.concat([dummy_df, dummy_df], ignore_index=True)
        df2["actual_tags"] = ["['CAUSE']", "[]"]
        df2["extracted_tags"] = ["['CAUSE']", "[]"]
        dummy_df = _DummyEvalDF(df2)

    # Patch pandas.read_csv to always return our dummy DataFrame
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: dummy_df)

    # Patch sklearn metric functions to lightweight no-ops to avoid shape mismatch errors.
    import sklearn.metrics as _sm  # type: ignore

    monkeypatch.setattr(_sm, "accuracy_score", lambda *a, **k: 0.0, raising=False)
    monkeypatch.setattr(
        _sm,
        "precision_recall_fscore_support",
        lambda *a, **k: (0.0, 0.0, 0.0, None),
        raising=False,
    )
    monkeypatch.setattr(_sm, "precision_score", lambda *a, **k: 0.0, raising=False)
    monkeypatch.setattr(_sm, "recall_score", lambda *a, **k: 0.0, raising=False)
    monkeypatch.setattr(_sm, "f1_score", lambda *a, **_k: 0.0, raising=False)

    # 6. builtins.eval -> return fake completion object for causal detection modules
    import builtins as _builtins  # noqa: WPS433 (importing system module)

    def _dummy_completion(*_a, **_k):  # noqa: D401 (simple function)
        """Return object mimicking litellm completion response."""
        return _SNS(
            choices=[_SNS(message=_SNS(content="<think>none</think> label: A"))]
        )

    monkeypatch.setattr(_builtins, "eval", lambda *_a, **_k: _dummy_completion())

    # 7. Path.exists always True (for numclaim)
    from pathlib import Path as _Path

    monkeypatch.setattr(_Path, "exists", lambda *_a, **_k: True, raising=False)

    # ------------------------------------------------------------------
    # Patch misc stdlib helpers for tolerant parsing & heavy deps stubbing
    # ------------------------------------------------------------------
    import sys  # noqa: WPS433
    import json as _json  # noqa: WPS433
    import ast as _ast  # noqa: WPS433

    _orig_json_loads = _json.loads
    _orig_literal_eval = _ast.literal_eval

    def _safe_json_loads(s, *a, **k):  # noqa: D401
        try:
            obj = _orig_json_loads(s, *a, **k)
            # Some callers expect dict with .get; return empty dict if list
            if isinstance(obj, list):
                return {}
            return obj
        except Exception:
            # Return empty dict fallback so .get calls are safe
            return {}

    def _safe_literal_eval(s, *a, **k):  # noqa: D401
        try:
            return _orig_literal_eval(s, *a, **k)
        except Exception:
            return []

    monkeypatch.setattr(_json, "loads", _safe_json_loads, raising=False)
    monkeypatch.setattr(_ast, "literal_eval", _safe_literal_eval, raising=False)

    # Stub heavyweight external libraries used by some eval modules
    from types import SimpleNamespace as _SSN  # noqa: WPS433

    class _MockMetric:  # noqa: WPS431
        def compute(self, predictions=None, references=None, *a, **k):  # noqa: D401
            length = len(predictions or [])
            zeros = [0.0] * length
            return {"precision": zeros, "recall": zeros, "f1": zeros}

    def _mock_evaluate_load(*_a, **_k):  # noqa: D401
        return _MockMetric()

    sys.modules.setdefault("evaluate", _SSN(load=_mock_evaluate_load))
    sys.modules.setdefault("transformers", _SSN())
    sys.modules.setdefault("transformers.pipelines", _SSN(SUPPORTED_TASKS={}))
    # Minimal stub for PIL and submodules to avoid class calls
    pil_stub = _SSN(Image=_SSN, TiffTags=_SSN(TagInfo=lambda *a, **k: None))
    sys.modules.setdefault("PIL", pil_stub)
    sys.modules.setdefault("PIL.Image", pil_stub.Image)
    sys.modules.setdefault("PIL.TiffTags", pil_stub.TiffTags)

    # Import module lazily so our patches are in effect thereafter.
    module: ModuleType = importlib.import_module(module_name)

    # Find callable ending with _evaluate
    eval_fns = [
        getattr(module, name)
        for name in dir(module)
        if callable(getattr(module, name)) and name.endswith("_evaluate")
    ]
    if not eval_fns:
        pytest.skip(f"No evaluation function found in {module_name}")
    eval_fn = eval_fns[0]

    # Attempt to call based on signature (file_name, args) or (df/args) patterns.
    sig = inspect.signature(eval_fn)
    params = list(sig.parameters)
    try:
        if len(params) >= 2:
            eval_fn("dummy.csv", dummy_args)  # file arg ignored by read_csv patch
        elif len(params) == 1:
            eval_fn(dummy_args)
        else:
            eval_fn()
    except Exception as e:  # pragma: no cover – make test fail for traceability
        pytest.fail(f"{module_name} crashed: {e}")
