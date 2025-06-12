"""Multi-task specific test configuration."""

import pytest


@pytest.fixture(autouse=True)
def mock_datasets_for_multitask(monkeypatch):
    """Ensure datasets are mocked for multi-task tests."""
    # Use the same _DummyRow and _DummyDataset from main conftest
    # to ensure consistency across all tests
    from tests.conftest import _DummyDataset

    # Mock at multiple levels to ensure it works in CI
    import importlib

    # 1. Mock the datasets module
    try:
        datasets = importlib.import_module("datasets")
        monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _DummyDataset())
    except ImportError:
        pass

    # 2. Mock in dataset_utils specifically
    try:
        from flame.utils import dataset_utils

        monkeypatch.setattr(
            dataset_utils, "load_dataset", lambda *a, **k: _DummyDataset()
        )
    except ImportError:
        pass

    # 3. Mock safe_load_dataset directly
    try:

        def mock_safe_load(*args, **kwargs):
            split = kwargs.get("split")
            dummy_data = _DummyDataset()
            if split:
                return dummy_data[split]
            return dummy_data

        monkeypatch.setattr(
            "flame.utils.dataset_utils.safe_load_dataset", mock_safe_load
        )
    except ImportError:
        pass
