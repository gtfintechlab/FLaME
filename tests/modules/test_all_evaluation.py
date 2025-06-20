"""Parametrised smoke-tests for every *_evaluate.py module.
Ensures evaluation code executes offline with stubbed data and metrics.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


# Custom replacement for SimpleNamespace to avoid conflicts
class _MockNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "flame" / "code"


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
        # Banking77 specific - use valid numeric labels
        "actual_labels_banking77": [11],  # card_arrival
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


@pytest.mark.modules
@pytest.mark.essential
@pytest.mark.parametrize("module_name", _discover_eval_modules())
def test_evaluation_module(module_name: str, dummy_args, monkeypatch):  # noqa: D103 – pytest test fn
    # Skip modules that are not in current release
    skip_prefixes = {
        "flame.code.mmlu",  # Not included in current release - deferred for future implementation
        "flame.code.econlogicqa",  # Not included in current release - deferred for future implementation
    }
    if any(module_name.startswith(p) for p in skip_prefixes):
        pytest.skip(f"Skipping {module_name} - not in current release")

    # Patch evaluate module EARLY to prevent heavy dependency imports
    import sys

    class MockBERTScore:
        """Mock BERTScore metric to avoid having to install bert-score and transformers for testing."""

        def compute(self, predictions, references, **kwargs):
            # Return list of scores matching the input length
            n = len(predictions) if predictions else 1
            return {"precision": [0.85] * n, "recall": [0.83] * n, "f1": [0.84] * n}

    class MockEvaluateModule:
        @staticmethod
        def load(metric_name, *args, **kwargs):
            if metric_name == "bertscore":
                return MockBERTScore()

            # Return a generic mock for other metrics
            class GenericMock:
                def compute(self, **kwargs):
                    return {"score": 0.5}

            return GenericMock()

    # Insert mock module to prevent actual import
    import types

    mock_evaluate = types.ModuleType("evaluate")
    mock_evaluate.load = MockEvaluateModule.load
    sys.modules["evaluate"] = mock_evaluate

    # Ensure evaluate.load is properly mocked
    monkeypatch.setattr("evaluate.load", MockEvaluateModule.load, raising=False)

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
    # For banking77, use numeric labels
    elif "banking77" in module_name:
        dummy_df["actual_labels"] = [
            11
        ]  # Use numeric label for banking77 (card_arrival)
        dummy_df["llm_responses"] = ["card_arrival"]  # Use valid banking77 label name
        dummy_df["extracted_labels"] = [-1]  # Will be populated by evaluation
    # For convfinqa and tatqa, ensure proper label types
    elif "convfinqa" in module_name:
        dummy_df["actual_labels"] = ["answer"]  # Non-None string
        dummy_df["llm_responses"] = ["answer"]  # Match the label type
    elif "tatqa" in module_name:
        dummy_df["actual_labels"] = [1]  # Numeric label
        dummy_df["llm_responses"] = ["answer"]  # String response
    elif "numclaim" in module_name:
        # NumClaim is binary classification - need both classes for sklearn
        df2 = pd.concat([dummy_df, dummy_df], ignore_index=True)
        df2["actual_labels"] = ["INCLAIM", "NOTCLAIM"]  # Will be mapped to 1 and 0
        df2["llm_responses"] = ["INCLAIM response", "NOTCLAIM response"]
        dummy_df = _DummyEvalDF(df2)
    elif "fiqa_task2" in module_name:
        # Ensure proper columns exist for fiqa_task2 with multiple rows
        df2 = pd.concat([dummy_df, dummy_df], ignore_index=True)
        df2["llm_responses"] = ["test answer", "different answer"]
        df2["actual_answers"] = ["test answer", "test answer"]
        dummy_df = _DummyEvalDF(df2)

    # Patch pandas.read_csv to always return our dummy DataFrame
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: dummy_df)

    # Patch sklearn metric functions to lightweight no-ops to avoid shape mismatch errors.
    import sklearn.metrics as _sm  # type: ignore
    import sklearn.metrics._classification as _smc  # type: ignore
    import sklearn.utils.multiclass as _sum  # type: ignore

    # Mock multilabel_confusion_matrix to avoid ndim attribute access
    import numpy as np

    monkeypatch.setattr(
        _smc,
        "multilabel_confusion_matrix",
        lambda *a, **k: np.array([[[0, 0], [0, 0]]]),
        raising=False,
    )

    # Create smarter mocks that handle binary classification edge cases
    def mock_precision_score(y_true, y_pred, average=None, **kwargs):
        # Just return a dummy value regardless of parameters
        return 0.85

    def mock_recall_score(y_true, y_pred, average=None, **kwargs):
        return 0.83

    def mock_f1_score(y_true, y_pred, average=None, **kwargs):
        return 0.84

    monkeypatch.setattr(_sm, "accuracy_score", lambda *a, **k: 0.9, raising=False)
    monkeypatch.setattr(
        _sm,
        "precision_recall_fscore_support",
        lambda *a, **k: (0.85, 0.83, 0.84, None),
        raising=False,
    )
    monkeypatch.setattr(_sm, "precision_score", mock_precision_score, raising=False)
    monkeypatch.setattr(_sm, "recall_score", mock_recall_score, raising=False)
    monkeypatch.setattr(_sm, "f1_score", mock_f1_score, raising=False)

    # Patch classification_report to avoid label validation
    def _mock_classification_report(y_true, y_pred, **kwargs):
        labels = kwargs.get(
            "labels", ["B-CAUSE", "I-CAUSE", "B-EFFECT", "I-EFFECT", "O"]
        )
        report = "              precision    recall  f1-score   support\n\n"
        for label in labels:
            report += f"{label:>12}      0.00      0.00      0.00         0\n"
        report += "\n    accuracy                           0.00         0\n"
        report += "   macro avg      0.00      0.00      0.00         0\n"
        report += "weighted avg      0.00      0.00      0.00         0\n"
        return report

    monkeypatch.setattr(
        _sm, "classification_report", _mock_classification_report, raising=False
    )

    # Patch sklearn utilities to avoid type checking issues
    monkeypatch.setattr(
        _smc,
        "_check_targets",
        lambda *a, **k: ("binary", [0, 1], [0, 1]),
        raising=False,
    )
    monkeypatch.setattr(_sum, "type_of_target", lambda *a, **k: "binary", raising=False)
    monkeypatch.setattr(_sum, "is_multilabel", lambda *a, **k: False, raising=False)

    # Patch _check_set_wise_labels to handle binary classification properly
    def mock_check_set_wise_labels(y_true, y_pred, average, labels=None, pos_label=1):
        # Just return labels that won't cause issues
        if average == "binary":
            return [0, 1]
        # Handle case where labels is already provided (might be numpy array)
        if labels is not None:
            return labels
        return [0, 1]

    monkeypatch.setattr(
        _smc, "_check_set_wise_labels", mock_check_set_wise_labels, raising=False
    )

    # 6. builtins.eval -> return fake completion object for causal detection modules
    import builtins as _builtins

    def _dummy_completion(*_a, **_k):  # noqa: D401 (simple function)
        """Return object mimicking litellm completion response."""
        # Check if this is for numclaim by looking at the messages
        if _a and isinstance(_a[0], list) and _a[0] and "INCLAIM" in str(_a[0]):
            # Return INCLAIM for numclaim
            content = "INCLAIM"
        else:
            # Default response for other modules
            content = "<think>none</think> label: A"

        return _MockNamespace(
            choices=[_MockNamespace(message=_MockNamespace(content=content))]
        )

    monkeypatch.setattr(_builtins, "eval", lambda *_a, **_k: _dummy_completion())

    # 7. Path.exists always True (for numclaim)
    from pathlib import Path as _Path

    monkeypatch.setattr(_Path, "exists", lambda *_a, **_k: True, raising=False)

    # 8. Patch process_batch_with_retry for module-specific responses
    from flame.utils import batch_utils

    def _module_specific_batch_retry(args, messages_batch, *_a, **_k):
        responses = []
        for i, msg in enumerate(messages_batch):
            msg_str = str(msg)
            # Return appropriate content based on module/content
            if "numclaim" in module_name:
                # Alternate between INCLAIM and other responses for binary classification
                content = "INCLAIM" if i % 2 == 0 else "NOT_INCLAIM"
            elif "INCLAIM" in msg_str:
                content = "INCLAIM"
            elif "fiqa" in module_name:
                content = "test answer"
            elif "banking77" in module_name:
                content = "card_arrival"
            else:
                content = "mock reply"

            responses.append(
                _MockNamespace(
                    choices=[_MockNamespace(message=_MockNamespace(content=content))]
                )
            )
        return responses

    monkeypatch.setattr(
        batch_utils,
        "process_batch_with_retry",
        _module_specific_batch_retry,
        raising=False,
    )

    # 8. Evaluate module already patched at the beginning of the test

    # ------------------------------------------------------------------
    # Patch misc stdlib helpers for tolerant parsing & heavy deps stubbing
    # ------------------------------------------------------------------
    import ast as _ast
    import json as _json
    import sys

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
    from types import SimpleNamespace as _SSN

    # Re-ensure evaluate module is mocked (in case it got imported)
    if "evaluate" in sys.modules:
        # Patch the load function directly
        import evaluate

        monkeypatch.setattr(evaluate, "load", MockEvaluateModule.load, raising=False)

    # Patch get_bertscore functions directly for ectsum and edtsum
    if "ectsum" in module_name:
        monkeypatch.setattr(
            "flame.code.ectsum.ectsum_evaluate.get_bertscore",
            lambda: MockBERTScore(),
            raising=False,
        )
    elif "edtsum" in module_name:
        monkeypatch.setattr(
            "flame.code.edtsum.edtsum_evaluate.get_bertscore",
            lambda: MockBERTScore(),
            raising=False,
        )

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
