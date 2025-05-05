import pytest
from main import run_tasks

@pytest.mark.parametrize("tasks", [
    ["numclaim", "finer"],
    ["ectsum"],
])
def test_multitask_inference(dummy_args, tasks):
    """
    Smoke-test the multi-task inference runner with 2 tasks and 1 task.
    Dummy fixtures stub I/O, so we only check for no exceptions.
    """
    dummy_args.mode = "inference"
    # ensure we use tasks list not dataset
    dummy_args.dataset = None
    dummy_args.tasks = tasks
    run_tasks(tasks, dummy_args.mode, dummy_args)
