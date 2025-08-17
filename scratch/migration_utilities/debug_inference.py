#!/usr/bin/env python3
"""Debug the inference pipeline to see what's happening with samples."""

import sys

sys.path.insert(0, "benchforge")

from bench_forge.flame.tasks.fomc import FOMCTask, FOMCConfig
from bench_forge.data.loader import HuggingFaceLoader, LoaderConfig

# Create task
config = FOMCConfig(
    name="fomc", huggingface_dataset="gtfintechlab/fomc_communication", num_samples=3
)
task = FOMCTask(config)

loader_config = LoaderConfig()
loader = HuggingFaceLoader(config=loader_config)
dataset = loader.load("gtfintechlab/fomc_communication", split="test")
print(f"Loaded {len(dataset) if hasattr(dataset, '__len__') else 'unknown'} samples")

# Check what prepare_prompts does
prompts = task.prepare_prompts(dataset)
print(f"\nPrepared {len(prompts)} prompts")

for i, prompt_data in enumerate(prompts[:3]):
    print(f"\n--- Prompt {i} ---")
    print(f"Sample keys: {list(prompt_data['metadata']['sample'].keys())}")
    print(
        f"Sample sentence: {prompt_data['metadata']['sample'].get('sentence', 'NOT FOUND')[:50]}..."
    )
    print(
        f"Prompt text: {prompt_data['prompt'][150:250]}..."
    )  # Show middle part of prompt
