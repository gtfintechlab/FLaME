"""Bridge module for migrating FLAME tasks to the bench-forge.

This module demonstrates how to adapt existing FLAME tasks to work with
the new bench-forge framework.
"""

import sys
from pathlib import Path
from typing import Callable

# Add FLAME to path for imports
flame_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(flame_root))

# Import Bench Forge components
from bench_forge import get_registry, register_task  # noqa: E402


class FLAMETaskAdapter:
    """Adapter to bridge FLAME tasks to the bench-forge framework."""

    def __init__(self, task_name: str):
        """Initialize the adapter for a specific FLAME task.

        Args:
            task_name: Name of the FLAME task to adapt
        """
        self.task_name = task_name
        self.inference_fn = None
        self.evaluation_fn = None

    def adapt_inference(self, flame_inference_fn: Callable) -> Callable:
        """Adapt a FLAME inference function to harness interface.

        Args:
            flame_inference_fn: Original FLAME inference function

        Returns:
            Adapted function for harness
        """

        def harness_inference(args):
            # FLAME functions expect args directly
            # Harness passes the same args structure
            return flame_inference_fn(args)

        return harness_inference

    def adapt_evaluation(self, flame_eval_fn: Callable) -> Callable:
        """Adapt a FLAME evaluation function to harness interface.

        Args:
            flame_eval_fn: Original FLAME evaluation function

        Returns:
            Adapted function for harness
        """

        def harness_evaluation(file_name, args):
            # FLAME evaluation functions have the same signature
            return flame_eval_fn(file_name, args)

        return harness_evaluation

    def register(self, inference_fn: Callable = None, evaluation_fn: Callable = None):
        """Register adapted FLAME task with the harness.

        Args:
            inference_fn: FLAME inference function
            evaluation_fn: FLAME evaluation function
        """
        if inference_fn:
            self.inference_fn = self.adapt_inference(inference_fn)

        if evaluation_fn:
            self.evaluation_fn = self.adapt_evaluation(evaluation_fn)

        # Register with harness
        register_task(
            self.task_name,
            inference_fn=self.inference_fn,
            evaluation_fn=self.evaluation_fn,
        )


def migrate_flame_task_maps():
    """Migrate all FLAME task mappings to the harness registry.

    This function imports FLAME's task maps and registers all tasks
    with the bench-forge.
    """
    try:
        # Import FLAME task registry
        from flame.task_registry import INFERENCE_MAP, EVALUATE_MAP

        # Get harness registry
        registry = get_registry()

        # Migrate inference tasks
        for task_name, inference_fn in INFERENCE_MAP.items():
            if not registry.has_task(task_name, "inference"):
                adapter = FLAMETaskAdapter(task_name)
                adapter.register(inference_fn=inference_fn)
                print(f"✓ Migrated inference task: {task_name}")

        # Migrate evaluation tasks
        for task_name, eval_fn in EVALUATE_MAP.items():
            if not registry.has_task(task_name, "evaluation"):
                # Check if we already have an adapter
                if registry.has_task(task_name, "inference"):
                    # Update existing registration
                    registry.register_evaluation(task_name, eval_fn)
                else:
                    # Create new adapter
                    adapter = FLAMETaskAdapter(task_name)
                    adapter.register(evaluation_fn=eval_fn)
                print(f"✓ Migrated evaluation task: {task_name}")

        # List migrated tasks
        print("\nMigration Summary:")
        print(f"Inference tasks: {len(registry.list_tasks('inference'))}")
        print(f"Evaluation tasks: {len(registry.list_tasks('evaluation'))}")

        return True

    except ImportError as e:
        print(f"Could not import FLAME modules: {e}")
        print("Make sure FLAME is properly installed")
        return False


def create_compatibility_layer():
    """Create a compatibility layer for FLAME code to use harness.

    This function creates import aliases so existing FLAME code
    can use harness utilities without modification.
    """
    import sys

    # Create module aliases
    class ModuleAlias:
        """Module alias for compatibility."""

        def __init__(self, harness_module):
            self.module = harness_module

        def __getattr__(self, name):
            return getattr(self.module, name)

    # Import harness modules
    from bench_forge.utils import batch, output, logging
    from bench_forge.core import registry, executor
    from bench_forge.config import config

    # Create FLAME-compatible module structure
    sys.modules["flame.utils.batch_utils"] = ModuleAlias(batch)
    sys.modules["flame.utils.output_utils"] = ModuleAlias(output)
    sys.modules["flame.utils.logging_utils"] = ModuleAlias(logging)
    sys.modules["flame.task_registry"] = ModuleAlias(registry)
    sys.modules["flame.code.inference"] = ModuleAlias(executor)
    sys.modules["flame.code.evaluate"] = ModuleAlias(executor)
    sys.modules["flame.config"] = ModuleAlias(config)

    print("✓ Compatibility layer created")
    print("  FLAME imports will now use harness modules")


def example_migration():
    """Example of migrating a specific FLAME task."""

    # Example: Migrate FOMC task
    try:
        from flame.code.fomc.fomc_inference import fomc_inference
        from flame.code.fomc.fomc_evaluate import fomc_evaluate

        # Create adapter
        adapter = FLAMETaskAdapter("fomc")
        adapter.register(inference_fn=fomc_inference, evaluation_fn=fomc_evaluate)

        print("✓ Successfully migrated FOMC task")

        # Now you can use it with the harness:
        # bench-forge --mode inference --tasks fomc --model "your-model"

    except ImportError:
        print("Could not import FOMC task - using mock example")

        # Mock example for demonstration
        def mock_fomc_inference(args):
            import pandas as pd

            return pd.DataFrame({"result": ["mock"]})

        def mock_fomc_evaluate(file_name, args):
            import pandas as pd

            return pd.DataFrame(), pd.DataFrame({"accuracy": [0.5]})

        adapter = FLAMETaskAdapter("mock_fomc")
        adapter.register(
            inference_fn=mock_fomc_inference, evaluation_fn=mock_fomc_evaluate
        )

        print("✓ Registered mock FOMC task for demonstration")


if __name__ == "__main__":
    print("FLAME to Eval-Harness Migration Bridge")
    print("=" * 40)

    # Example 1: Migrate all FLAME tasks
    print("\n1. Attempting to migrate all FLAME tasks...")
    success = migrate_flame_task_maps()

    if not success:
        print("\n2. Running example migration with mock task...")
        example_migration()

    # Example 2: Create compatibility layer
    print("\n3. Creating compatibility layer...")
    create_compatibility_layer()

    print("\n✅ Migration bridge setup complete!")
    print("\nYou can now:")
    print("1. Use bench-forge CLI with FLAME tasks")
    print("2. Import FLAME modules that will use harness utilities")
    print("3. Gradually migrate FLAME code to use harness directly")
