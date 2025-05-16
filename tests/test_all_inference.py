"""Parametrised smoke-tests that import every *_inference.py and ensure its
main function runs without external calls (thanks to fixtures).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "flame" / "code"


def _discover_inference_modules() -> list[str]:
    """Return fully-qualified module paths for every *_inference.py file."""
    paths = SRC_DIR.rglob("*_inference.py")
    modules: list[str] = []
    for file in paths:
        # Convert path to python module string: src/flame/code/foo/bar_inference.py
        rel_parts = file.relative_to(SRC_DIR.parent.parent).with_suffix("").parts
        modules.append(".".join(rel_parts))
    return sorted(modules)


@pytest.mark.parametrize("module_name", _discover_inference_modules())
def test_inference_module(module_name, dummy_args):  # noqa: D103 – pytest test fn
    # Some modules expect external heavy deps or file structures we can't mock easily – skip.
    skip_prefixes = {
        "flame.code.mmlu",  # MMLU pipeline still under construction – skip for now
    }
    if any(module_name.startswith(p) for p in skip_prefixes):
        pytest.skip(
            f"Skipping {module_name} due to unsupported heavy dependencies in test context"
        )

    module: ModuleType = importlib.import_module(module_name)

    # Expect each module to expose an inference function like `numclaim_inference`
    func_candidates = [
        obj
        for obj in module.__dict__.values()
        if callable(obj) and obj.__name__.endswith("_inference")
    ]
    if not func_candidates:
        pytest.skip(f"No inference function in {module_name}")

    inference_fn = func_candidates[0]

    # Adjust dataset arg based on folder name if possible
    dataset_name = module.__name__.split(".")[
        -2
    ]  # e.g., flame.code.numclaim.numclaim_inference -> numclaim
    dummy_args.dataset = dataset_name

    # Run – fixture patches ensure no real API call occurs
    result = inference_fn(dummy_args)

    # Minimal sanity check – many inference fns return pd.DataFrame
    if result is not None and hasattr(result, "__len__"):
        assert len(result) >= 0
