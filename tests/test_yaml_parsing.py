import pytest
import sys
import yaml
from main import parse_arguments


def write_temp_yaml(tmp_path, data):
    file = tmp_path / "config.yml"
    file.write_text(yaml.dump(data))
    return str(file)

@pytest.mark.parametrize("cfg,cli_args,expected", [
    ( {"tasks": ["numclaim", "finer"]}, [], ["numclaim", "finer"] ),
    ( {"tasks": ["numclaim"]}, ["--tasks", "ectsum"], ["ectsum"] ),
])
def test_tasks_merge(tmp_path, monkeypatch, cfg, cli_args, expected):
    cfg_path = write_temp_yaml(tmp_path, cfg)
    argv = ["main.py", "--config", cfg_path, "--mode", "inference"] + cli_args
    monkeypatch.setattr(sys, 'argv', argv)
    args = parse_arguments()
    assert args.tasks == expected


def test_cli_overrides_model(tmp_path, monkeypatch):
    cfg_path = write_temp_yaml(tmp_path, {"model": "m1"})
    argv = ["main.py", "--config", cfg_path, "--mode", "inference", "--model", "m2"]
    monkeypatch.setattr(sys, 'argv', argv)
    args = parse_arguments()
    assert args.model == "m2"


def test_defaults_are_applied(monkeypatch):
    # No config, minimal CLI
    # Set up minimal CLI args with no config
    monkeypatch.setattr(__import__('sys'), 'argv', ["main.py", "--mode", "inference", "--tasks", "numclaim"])
    args = parse_arguments()
    assert args.max_tokens == 128
    assert args.temperature == 0.0
    assert args.top_p == 0.9
    assert args.repetition_penalty == 1.0
    assert args.batch_size == 10
    assert args.prompt_format == "zero_shot"