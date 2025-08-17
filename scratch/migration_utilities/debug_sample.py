#!/usr/bin/env python3
"""Debug what samples are being passed to create_prompt."""

import sys
import json

sys.path.insert(0, "benchforge")

from bench_forge.flame.tasks.fomc import FOMCTask, FOMCConfig
from datasets import load_dataset

# Load dataset
dataset = load_dataset("gtfintechlab/fomc_communication", split="test")
print(f"Dataset has {len(dataset)} samples")

# Check first sample
sample = dataset[0]
print("\nFirst sample keys:", list(sample.keys()))
print("First sample:", json.dumps(sample, indent=2))

# Create task
config = FOMCConfig(name="fomc")
task = FOMCTask(config)

print(f"\nConfig text_field: {config.text_field}")
print(f"Config label_field: {config.label_field}")

# Test prompt creation
prompt = task.create_prompt(sample)
print("\nGenerated prompt:")
print(prompt[:200] + "...")
