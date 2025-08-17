#!/usr/bin/env python3
"""Debug which process_responses is being called."""

import sys

sys.path.insert(0, "benchforge")

from bench_forge.flame.tasks.fomc import FOMCTask, FOMCConfig

# Create task
config = FOMCConfig(name="fomc")
task = FOMCTask(config)

# Check methods
print("Methods on FOMCTask:")
print(f"  process_responses: {task.process_responses}")
print(f"  format_results: {task.format_results}")
print(f"  extract_label_from_response: {task.extract_label_from_response}")

# Check MRO
print("\nMethod Resolution Order:")
for cls in FOMCTask.__mro__:
    print(f"  {cls}")
    if hasattr(cls, "process_responses"):
        print("    -> has process_responses")
