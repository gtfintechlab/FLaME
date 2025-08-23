import importlib.util
import pathlib
import sys
import types

import pytest


# Stub external dependencies to allow importing TATQATask without full package installation
sys.modules.setdefault("pandas", types.SimpleNamespace(DataFrame=object))

bench_forge = types.ModuleType("bench_forge")
sys.modules["bench_forge"] = bench_forge

flame_mod = types.ModuleType("bench_forge.flame")
adapter_mod = types.ModuleType("bench_forge.flame.adapter")


class _DummyConfig:
    pass


def flame_task(name):
    def decorator(cls):
        return cls

    return decorator


adapter_mod.FLAMEConfig = _DummyConfig
adapter_mod.FLAMETask = object
adapter_mod.flame_task = flame_task

sys.modules["bench_forge.flame"] = flame_mod
sys.modules["bench_forge.flame.adapter"] = adapter_mod
bench_forge.flame = flame_mod

tasks_mod = types.ModuleType("bench_forge.tasks")
config_mod = types.ModuleType("bench_forge.tasks.config")


class PromptFormat:
    ZERO_SHOT = 0
    FEW_SHOT = 1


config_mod.PromptFormat = PromptFormat
tasks_mod.config = config_mod

sys.modules["bench_forge.tasks"] = tasks_mod
sys.modules["bench_forge.tasks.config"] = config_mod
bench_forge.tasks = tasks_mod

# Load module directly from file path
tatqa_path = (
    pathlib.Path(__file__).resolve().parents[2]
    / "vendor/benchforge/bench_forge/flame/tasks/tatqa.py"
)
spec = importlib.util.spec_from_file_location("tatqa", tatqa_path)
tatqa = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(tatqa)  # type: ignore[assignment]
TATQATask = tatqa.TATQATask


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The revenue was $5 million.", "5000000"),
        ("The investment reached 3 billion dollars", "3000000000"),
        ("Net worth stands at $2.5 trillion", "2500000000000"),
        ("Funding totaled $4M last year", "4000000"),
        ("The deal was valued at 7B dollars", "7000000000"),
        ("Budget was $6T", "6000000000000"),
    ],
)
def test_extract_currency_with_magnitude(text: str, expected: str) -> None:
    task = object.__new__(TATQATask)
    assert task._extract_currency(text) == expected
