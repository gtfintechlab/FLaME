#!/usr/bin/env python3
"""Test script to verify FLAME-BenchForge FOMC integration."""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all necessary imports work."""
    logger.info("Testing imports...")

    try:
        # Core FLAME imports
        from flame import benchforge

        logger.info("✓ flame.benchforge imported")

        # Check BenchForge availability
        if not benchforge.BENCHFORGE_AVAILABLE:
            logger.error("✗ BenchForge not available!")
            return False
        logger.info("✓ BenchForge is available")

        # Import FLAME tasks
        from flame.tasks import register_all_flame_tasks  # noqa: F401

        logger.info("✓ flame.tasks imported")

        # Import FOMC task specifically
        from flame.tasks.fomc import FOMCTask  # noqa: F401

        logger.info("✓ FOMCTask imported")

        return True

    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False


def test_task_registration():
    """Test that FOMC task can be registered."""
    logger.info("\nTesting task registration...")

    try:
        from flame.tasks import register_all_flame_tasks
        from flame.benchforge import get_registry

        # Register tasks
        registered = register_all_flame_tasks()
        logger.info(f"✓ Registered {len(registered)} tasks: {registered}")

        # Verify with registry
        registry = get_registry()
        all_tasks = registry.list_tasks()
        logger.info(f"✓ Total tasks in registry: {len(all_tasks)}")

        if "fomc" in all_tasks:
            logger.info("✓ FOMC task successfully registered")
            return True
        else:
            logger.warning("⚠ FOMC task not found in registry")
            return False

    except Exception as e:
        logger.error(f"✗ Registration error: {e}")
        return False


def test_fomc_task_creation():
    """Test creating and configuring FOMC task."""
    logger.info("\nTesting FOMC task creation...")

    try:
        from flame.tasks import create_task
        from flame.benchforge import FLAMEConfig, PromptFormat

        # Create config
        config = FLAMEConfig(
            name="fomc",
            dataset="fomc",
            huggingface_dataset="gtfintechlab/fomc_communication",
            prompt_format=PromptFormat.ZERO_SHOT,
            batch_size=5,
            max_tokens=10,
        )
        logger.info("✓ Created FLAMEConfig")

        # Create task
        task = create_task("fomc", config)
        logger.info("✓ Created FOMC task instance")

        # Test prompt creation
        sample = {
            "text": "The Committee decided to raise the federal funds rate by 25 basis points.",
            "label": "HAWKISH",
        }
        prompt = task.create_prompt(sample)
        logger.info("✓ Generated prompt successfully")
        logger.info(f"  Sample prompt (first 100 chars): {prompt[:100]}...")

        return True

    except Exception as e:
        logger.error(f"✗ Task creation error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_inference_engine():
    """Test creating inference engine."""
    logger.info("\nTesting inference engine creation...")

    try:
        from flame.benchforge import create_inference_engine, create_llm_client
        from flame.benchforge import LLMConfig

        # Create LLM config (mock mode for testing)
        llm_config = LLMConfig(
            provider="litellm",
            model="together_ai/meta-llama/Llama-3-8B-Instruct",
            max_tokens=10,
            temperature=0.0,
        )

        # Note: This will fail without API keys, but we can test the creation
        try:
            create_llm_client(config=llm_config)
            logger.info("✓ Created LLM client")
        except Exception as e:
            logger.warning(
                f"⚠ LLM client creation failed (expected without API keys): {e}"
            )

        # Create inference engine
        create_inference_engine(output_dir=Path("test_results"))
        logger.info("✓ Created inference engine")

        return True

    except Exception as e:
        logger.error(f"✗ Inference engine error: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("FLAME-BenchForge FOMC Integration Test")
    logger.info("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Task Registration", test_task_registration),
        ("FOMC Task Creation", test_fomc_task_creation),
        ("Inference Engine", test_inference_engine),
    ]

    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary:")
    logger.info("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n✅ All tests passed!")
    else:
        logger.info("\n❌ Some tests failed. Please review the output above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
