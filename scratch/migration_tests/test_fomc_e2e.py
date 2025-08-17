#!/usr/bin/env python3
"""E2E test for FOMC task using FLAME-BenchForge integration.

This test simulates a real user workflow:
1. Load FOMC dataset
2. Run inference (with mock LLM for testing)
3. Evaluate results
4. Generate report
"""

import sys
import logging
from pathlib import Path
from unittest.mock import Mock

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_mock_llm():
    """Setup mock LLM responses for testing."""
    # Create predictable responses for testing
    responses = {
        "hawkish": "Classification: HAWKISH",
        "dovish": "Classification: DOVISH",
        "neutral": "Classification: NEUTRAL",
    }

    def mock_complete(prompt):
        """Mock LLM completion."""
        prompt_lower = prompt.lower()
        if "raise" in prompt_lower or "tightening" in prompt_lower:
            return responses["hawkish"]
        elif "accommodative" in prompt_lower or "stimulus" in prompt_lower:
            return responses["dovish"]
        else:
            return responses["neutral"]

    def mock_complete_batch(prompts, **kwargs):
        """Mock batch completion."""
        # Ignore extra kwargs like show_progress
        return [mock_complete(p) for p in prompts]

    return mock_complete, mock_complete_batch


def test_fomc_data_loading():
    """Test loading FOMC dataset."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing FOMC Data Loading")
    logger.info("=" * 60)

    try:
        from flame.benchforge import FLAMEConfig
        from flame.tasks import create_task

        # Create FOMC task
        config = FLAMEConfig(
            name="fomc",
            dataset="fomc",
            huggingface_dataset="gtfintechlab/fomc_communication",
            num_samples=5,  # Just test with 5 samples
        )

        task = create_task("fomc", config)
        logger.info("✓ Created FOMC task")

        # Load dataset
        dataset = task.load_dataset(split="test")
        logger.info(f"✓ Loaded dataset with {len(dataset)} samples")

        # Check first sample
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"✓ First sample has keys: {list(sample.keys())}")
            text_field = (
                config.text_field if config.text_field in sample else "sentence"
            )
            label_field = (
                config.label_field if config.label_field in sample else "hawkish_dovish"
            )
            logger.info(f"  Text preview: {str(sample.get(text_field, ''))[:100]}...")
            logger.info(f"  Label: {sample.get(label_field, 'N/A')}")

        return True, dataset, task

    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None, None


def test_fomc_inference(dataset, task):
    """Test FOMC inference with mock LLM."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing FOMC Inference")
    logger.info("=" * 60)

    try:
        from flame.benchforge import create_inference_engine, LLMConfig, LLMClient

        # Create mock LLM client
        mock_complete, mock_complete_batch = setup_mock_llm()

        # Create LLM config
        llm_config = LLMConfig(
            provider="litellm",
            model="mock-model",
            max_tokens=10,
            temperature=0.0,
        )

        # Create and mock the LLM client
        llm_client = LLMClient(llm_config)
        llm_client.complete = Mock(side_effect=mock_complete)
        llm_client.complete_batch = Mock(side_effect=mock_complete_batch)

        logger.info("✓ Created mock LLM client")

        # Create inference engine
        output_dir = Path("scratch/test_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        engine = create_inference_engine(llm_client=llm_client, output_dir=output_dir)
        logger.info("✓ Created inference engine")

        # Run inference on subset of data
        logger.info("Running inference on 5 samples...")

        # Use the task's config with num_samples set
        task.config.num_samples = 5
        task.config.batch_size = 2  # Small batch for testing

        result = engine.run(task="fomc", config=task.config)

        logger.info("✓ Inference completed:")
        logger.info(f"  - Processed {len(result.results_df)} samples")
        logger.info(f"  - Output saved to: {result.output_path}")
        if hasattr(result, "metadata") and "duration_seconds" in result.metadata:
            logger.info(
                f"  - Execution time: {result.metadata['duration_seconds']:.2f}s"
            )

        # Check results DataFrame
        df = result.results_df
        logger.info(f"✓ Results DataFrame columns: {list(df.columns)}")

        # Verify extractions
        if "extracted_response" in df.columns:
            extraction_rate = df["extracted_response"].notna().mean()
            logger.info(f"  - Extraction rate: {extraction_rate:.1%}")

            # Show sample predictions
            logger.info("  - Sample predictions:")
            for idx, row in df.head(3).iterrows():
                logger.info(
                    f"    {idx}: {row.get('extracted_response', 'N/A')} (GT: {row.get('ground_truth', 'N/A')})"
                )

        return True, result

    except Exception as e:
        logger.error(f"✗ Inference failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_fomc_evaluation(inference_result):
    """Test FOMC evaluation."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing FOMC Evaluation")
    logger.info("=" * 60)

    try:
        from flame.benchforge import create_evaluation_engine

        # Create evaluation engine
        eval_output_dir = Path("scratch/test_evaluations")
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        eval_engine = create_evaluation_engine(output_dir=eval_output_dir)
        logger.info("✓ Created evaluation engine")

        # Run evaluation
        eval_result = eval_engine.evaluate(
            results_path=inference_result.output_path,
            task="fomc",
            metrics=["accuracy", "f1_macro", "confusion_matrix"],
            save_results=True,
        )

        logger.info("✓ Evaluation completed:")
        logger.info(f"  - Task: {eval_result.task_name}")
        logger.info(f"  - Model: {eval_result.model}")
        logger.info(f"  - Samples: {eval_result.num_samples}")

        # Display metrics
        logger.info("  - Metrics:")
        for metric, value in eval_result.metrics.items():
            if isinstance(value, float):
                logger.info(f"    {metric}: {value:.4f}")
            elif metric == "confusion_matrix" and hasattr(value, "shape"):
                logger.info(f"    {metric}: {value.shape} matrix")
            else:
                logger.info(f"    {metric}: {type(value).__name__}")

        if eval_result.num_errors > 0:
            logger.warning(f"  - Errors: {eval_result.num_errors}")

        return True, eval_result

    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_report_generation(eval_result):
    """Test report generation."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Report Generation")
    logger.info("=" * 60)

    try:
        # Create a simple report
        report_path = Path("scratch/test_report.md")

        report = f"""# FOMC Task Evaluation Report

## Summary
- **Task**: {eval_result.task_name}
- **Model**: {eval_result.model}
- **Date**: {eval_result.timestamp}
- **Samples**: {eval_result.num_samples}

## Performance Metrics
"""

        for metric, value in eval_result.metrics.items():
            if isinstance(value, float):
                report += f"- **{metric}**: {value:.4f}\n"

        report += f"""
## Files Generated
- Evaluation output: {getattr(eval_result, "output_path", "scratch/test_evaluations/")}

## Status
✅ E2E test completed successfully
"""

        report_path.write_text(report)
        logger.info(f"✓ Report generated: {report_path}")
        logger.info(f"  Report preview:\n{report[:500]}...")

        return True

    except Exception as e:
        logger.error(f"✗ Report generation failed: {e}")
        return False


def main():
    """Run complete E2E test."""
    logger.info("=" * 60)
    logger.info("FLAME-BenchForge FOMC E2E Test")
    logger.info("=" * 60)

    # Register tasks
    from flame.tasks import register_all_flame_tasks

    registered = register_all_flame_tasks()
    logger.info(f"Registered tasks: {registered}")

    # Run tests in sequence
    success = True

    # 1. Data Loading
    data_success, dataset, task = test_fomc_data_loading()
    success = success and data_success

    if not data_success:
        logger.error("Data loading failed, cannot continue")
        return 1

    # 2. Inference
    inference_success, inference_result = test_fomc_inference(dataset, task)
    success = success and inference_success

    if not inference_success:
        logger.error("Inference failed, cannot continue")
        return 1

    # 3. Evaluation
    eval_success, eval_result = test_fomc_evaluation(inference_result)
    success = success and eval_success

    if not eval_success:
        logger.error("Evaluation failed, cannot continue")
        return 1

    # 4. Report Generation
    report_success = test_report_generation(eval_result)
    success = success and report_success

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("E2E Test Summary")
    logger.info("=" * 60)
    logger.info(f"Data Loading: {'✓' if data_success else '✗'}")
    logger.info(f"Inference: {'✓' if inference_success else '✗'}")
    logger.info(f"Evaluation: {'✓' if eval_success else '✗'}")
    logger.info(f"Report Generation: {'✓' if report_success else '✗'}")

    if success:
        logger.info("\n✅ All E2E tests passed!")
        logger.info("\nThe user can now run:")
        logger.info("  uv run flame --mode inference --task fomc --model <model_name>")
        logger.info(
            "  uv run flame --mode evaluate --task fomc --file_name <results_file>"
        )
        return 0
    else:
        logger.error("\n❌ Some E2E tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
