import pytest
import sys
import yaml
from main import parse_arguments

pytestmark = pytest.mark.unit


def write_temp_yaml(tmp_path, data):
    file = tmp_path / "config.yml"
    file.write_text(yaml.dump(data))
    return str(file)


@pytest.mark.parametrize(
    "cfg,cli_args,expected",
    [
        ({"tasks": ["numclaim", "finer"]}, [], ["numclaim", "finer"]),
        ({"tasks": ["numclaim"]}, ["--tasks", "ectsum"], ["ectsum"]),
    ],
)
def test_tasks_merge(tmp_path, monkeypatch, cfg, cli_args, expected):
    cfg_path = write_temp_yaml(tmp_path, cfg)
    argv = ["main.py", "--config", cfg_path, "--mode", "inference"] + cli_args
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_arguments()
    assert args.tasks == expected


def test_yaml_tasks_list_parsing(tmp_path, monkeypatch):
    """Test that tasks list in YAML is correctly parsed"""
    # Create a temporary YAML config
    config_data = {
        "mode": "inference",
        "tasks": ["fomc", "numclaim", "finer"],
        "model": "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Mock sys.argv to simulate command line args
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_file)])

    # Parse arguments
    args = parse_arguments()

    # Verify tasks are parsed correctly
    assert args.mode == "inference"
    assert args.tasks == ["fomc", "numclaim", "finer"]
    assert args.model == "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"


def test_cli_multiple_tasks_override(tmp_path, monkeypatch):
    """Test that multiple CLI tasks arguments override YAML config"""
    # Create a YAML config with tasks
    config_data = {
        "mode": "inference",
        "tasks": ["fomc", "numclaim", "finer"],
        "model": "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Mock sys.argv with CLI override for multiple tasks
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--config", str(config_file), "--tasks", "fpb", "finentity"],
    )

    # Parse arguments
    args = parse_arguments()

    # CLI tasks should override YAML
    assert args.tasks == ["fpb", "finentity"]


def test_yaml_with_all_parameters(tmp_path, monkeypatch):
    """Test YAML parsing with comprehensive config"""
    config_data = {
        "mode": "inference",
        "tasks": ["fomc", "numclaim", "finer"],
        "model": "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "max_tokens": 256,
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "batch_size": 5,
        "prompt_format": "few_shot",
    }

    config_file = tmp_path / "full_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Mock sys.argv
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_file)])
    args = parse_arguments()

    # Verify all parameters
    assert args.tasks == ["fomc", "numclaim", "finer"]
    assert args.mode == "inference"
    assert args.model == "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
    assert args.max_tokens == 256
    assert args.temperature == 0.1
    assert args.top_p == 0.95
    assert args.top_k == 50
    assert args.repetition_penalty == 1.2
    assert args.batch_size == 5
    assert args.prompt_format == "few_shot"


def test_single_dataset_backward_compatibility(tmp_path, monkeypatch):
    """Test backward compatibility with single dataset field"""
    # Create a YAML config with legacy 'dataset' field
    config_data = {
        "mode": "inference",
        "dataset": "fomc",
        "model": "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Mock sys.argv
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_file)])

    # Parse arguments
    args = parse_arguments()

    # Should have dataset set but no tasks
    assert args.dataset == "fomc"
    assert not hasattr(args, "tasks") or args.tasks is None


def test_evaluate_mode_with_tasks(tmp_path, monkeypatch):
    """Test multi-task evaluation mode"""
    config_data = {
        "mode": "evaluate",
        "tasks": ["numclaim", "finer"],
        "file_name": "test_results.csv",
    }

    config_file = tmp_path / "eval_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Mock sys.argv
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_file)])

    # Parse arguments
    args = parse_arguments()

    # Verify evaluate mode parameters
    assert args.mode == "evaluate"
    assert args.tasks == ["numclaim", "finer"]
    assert args.file_name == "test_results.csv"


def test_cli_overrides_model(tmp_path, monkeypatch):
    cfg_path = write_temp_yaml(tmp_path, {"model": "m1"})
    argv = ["main.py", "--config", cfg_path, "--mode", "inference", "--model", "m2"]
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_arguments()
    assert args.model == "m2"


def test_defaults_are_applied(monkeypatch):
    # No config, minimal CLI
    # Set up minimal CLI args with no config
    monkeypatch.setattr(
        __import__("sys"),
        "argv",
        ["main.py", "--mode", "inference", "--tasks", "numclaim"],
    )
    args = parse_arguments()
    assert args.max_tokens == 128
    assert args.temperature == 0.0
    assert args.top_p == 0.9
    assert args.repetition_penalty == 1.0
    assert args.batch_size == 10
    assert args.prompt_format == "zero_shot"
