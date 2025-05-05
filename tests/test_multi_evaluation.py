import pytest
import main

@pytest.mark.parametrize("tasks", [
    ["numclaim", "finer"],
    ["ectsum"],
])
def test_multitask_evaluation(monkeypatch, tasks):
    """
    Smoke-test the multi-task evaluation runner with 2 tasks and 1 task.
    Expect `evaluate` to be called with each task.
    """
    # Prepare dummy args
    from types import SimpleNamespace
    dummy_args = SimpleNamespace(mode="evaluate", tasks=tasks, file_name="dummy.csv")
    calls = []
    # Stub evaluate() in main module
    monkeypatch.setattr(main, 'evaluate', lambda args: calls.append(args.task))

    # Run multi-task evaluation
    main.run_tasks(tasks, dummy_args.mode, dummy_args)
    assert calls == tasks
