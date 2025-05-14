"""Global pytest fixtures ensuring FLaME tests run offline and fast.

Patched automatically for every test:
• litellm.completion / batch_completion -> static mock responses
• datasets.load_dataset -> small in-memory dataset
• time.sleep -> no-op
• RESULTS_DIR / LOG_DIR redirected to temp folder
"""

from __future__ import annotations

import time as _time
import importlib
from pathlib import Path
from types import SimpleNamespace as _SN

import pytest
import litellm

# Silence deprecation warnings (Pydantic, litellm deprecations)
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Mock LiteLLM helpers
# ---------------------------------------------------------------------------


class _FakeCompletion:
    def __init__(self, content: str = "mock reply"):
        self.choices = [_SN(message=_SN(content=content))]
        self.model = "mock-model"
        self.created = 0
        self.usage = _SN(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    def __repr__(self):
        return "<FakeCompletion mock reply>"


def _fake_completion(*_args, **_kwargs):
    return _FakeCompletion()


def _fake_batch_completion(*_args, **_kwargs):  # type: ignore
    # LiteLLM batch_completion signature often: (model, messages=[...])
    messages = _kwargs.get("messages")
    if messages is None and len(_args) >= 2:
        messages = _args[1]
    messages = messages or []
    return [_FakeCompletion() for _ in messages]


# ---------------------------------------------------------------------------
# Mock datasets helper
# ---------------------------------------------------------------------------


class _DummyRow(dict):
    _DEFAULTS = {
        # Generic keys
        "text": "dummy text",
        "document": "dummy document",
        "sentence": "dummy sentence",
        # FinRED specific
        "entities": [("ent_a", "ent_b")],
        "relations": ["rel_dummy"],
        # ReFinD specific
        "token": ["t1", "t2", "t3", "t4", "t5"],
        "e1_start": 0,
        "e1_end": 1,
        "e2_start": 3,
        "e2_end": 4,
        "rel_group": "rel_dummy",
    }

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        return self._DEFAULTS.get(key, f"dummy {key}")


class _DummyDataset(list):
    def __init__(self):
        super().__init__([_DummyRow(), _DummyRow()])

    def __getitem__(self, item):  # type: ignore[override]
        if item in {"train", "test", "validation"}:
            return self
        return super().__getitem__(item)


# ---------------------------------------------------------------------------
# Autouse fixture applying patches
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_external(monkeypatch, tmp_path_factory):
    # 1. LiteLLM
    monkeypatch.setattr(litellm, "completion", _fake_completion)
    monkeypatch.setattr(litellm, "batch_completion", _fake_batch_completion)

    # 2. datasets.load_dataset
    datasets = importlib.import_module("datasets")
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _DummyDataset())

    # 3. time.sleep
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)

    # 3b. flame.code.tokens.tokens -> returns empty list to avoid DeprecationError
    try:
        import flame.code.tokens as _tokens_mod

        monkeypatch.setattr(_tokens_mod, "tokens", lambda *_a, **_k: [])
    except ModuleNotFoundError:
        pass

    # 4. Together AI client
    try:
        import together

        class _MockCompletions:
            def create(self, *args, **kwargs):
                return _fake_completion()

        class _MockChat:
            def __init__(self):
                self.completions = _MockCompletions()

        class _MockTogether:
            def __init__(self, *args, **kwargs):
                self.chat = _MockChat()

        monkeypatch.setattr(together, "Together", _MockTogether)
    except ModuleNotFoundError:
        pass

    # 5. nltk.download – prevent network download
    try:
        import nltk

        monkeypatch.setattr(nltk, "download", lambda *_a, **_k: None)
    except ModuleNotFoundError:
        pass

    # 6. Redirect output dirs
    temp_root: Path = tmp_path_factory.mktemp("flame_artifacts")
    # Redirect results, logs, and evaluation outputs to temporary dirs
    results_dir = temp_root / "results"
    logs_dir = temp_root / "logs"
    eval_dir = temp_root / "evaluation_test_artifacts"
    for d in (results_dir, logs_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)
    try:
        import flame.config as _cfg

        monkeypatch.setattr(_cfg, "RESULTS_DIR", results_dir, raising=False)
        monkeypatch.setattr(_cfg, "LOG_DIR", logs_dir, raising=False)
        monkeypatch.setattr(_cfg, "EVALUATION_DIR", eval_dir, raising=False)
    except ModuleNotFoundError:
        pass

    # 6b. Stub MMLULoader to avoid heavy dataset logic
    try:
        import flame.code.mmlu.mmlu_loader as _mml

        class _StubMMLULoader:  # type: ignore
            def __init__(self, subjects=None, split="test", num_few_shot: int = 5):
                self.subjects = subjects or ["economics"]
                self.split = split
                self.num_few_shot = num_few_shot

            def load_few_shot_examples(self):
                return [
                    {"question": "q", "choices": ["a", "b", "c", "d"], "answer": "A"}
                ]

            def load(self):
                import pandas as _pd

                data = {
                    "subject": ["economics", "economics"],
                    "question": ["What?", "Why?"],
                    "choices": [["A", "B", "C", "D"]] * 2,
                    "answer": ["A", "B"],
                }
                df = _pd.DataFrame(data)
                return df, self.load_few_shot_examples()

            # Alias to match get_subjects_summary if needed
            def get_subjects_summary(self):
                return {"economics": 2}

        monkeypatch.setattr(_mml, "MMLULoader", _StubMMLULoader)
    except ModuleNotFoundError:
        pass

    # 7. Alias missing prompt names
    try:
        import flame.code.prompts_zeroshot as _pz

        if not hasattr(_pz, "tatqa_prompt") and hasattr(_pz, "tatqa_zeroshot_prompt"):
            _pz.tatqa_prompt = _pz.tatqa_zeroshot_prompt  # type: ignore
    except ModuleNotFoundError:
        pass

    yield


# ---------------------------------------------------------------------------
# Generic args fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def dummy_args():
    return _SN(
        model="gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=16,
        top_p=1.0,
        top_k=None,
        repetition_penalty=1.0,
        batch_size=2,
        prompt_format="zeroshot",
        dataset="dummy",
        mmlu_subjects=["economics"],
        mmlu_split="test",
        mmlu_num_few_shot=1,
        evaluation_batch_size=2,
        evaluation_max_tokens=5,
        evaluation_temperature=0.0,
        evaluation_top_k=None,
        evaluation_top_p=1.0,
        evaluation_repetition_penalty=1.0,
    )
