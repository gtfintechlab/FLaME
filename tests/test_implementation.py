#!/usr/bin/env python3
"""Quick test to verify sample_size implementation"""

from unittest.mock import patch, Mock
from main import parse_arguments

# Test 1: CLI arguments
print("Test 1: CLI arguments")
test_args = ["main.py", "--mode", "inference", "--model", "test_model", "--sample_size", "10", "--tasks", "fpb"]
with patch('sys.argv', test_args):
    args = parse_arguments()
    print(f"sample_size from CLI: {args.sample_size}")
    assert args.sample_size == 10

# Test 2: YAML config
print("\nTest 2: YAML config")
import tempfile
import yaml

config = {"model": "test_model", "sample_size": 15, "tasks": ["fpb"]}
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
    yaml.dump(config, f)
    f.flush()
    
    test_args = ["main.py", "--config", f.name, "--mode", "inference"]
    with patch('sys.argv', test_args):
        args = parse_arguments()
        print(f"sample_size from YAML: {args.sample_size}")
        assert args.sample_size == 15

# Test 3: Verify the hasattr check
print("\nTest 3: hasattr check")
args = Mock()
args.sample_size = 5

if hasattr(args, 'sample_size') and args.sample_size is not None:
    print(f"Has sample_size: {args.sample_size}")

# Test 4: Missing attribute
print("\nTest 4: Missing attribute")
args2 = Mock(spec=[])  # No attributes
if hasattr(args2, 'sample_size') and args2.sample_size is not None:
    print("Has sample_size")
else:
    print("No sample_size attribute or None")

print("\nAll tests passed!")