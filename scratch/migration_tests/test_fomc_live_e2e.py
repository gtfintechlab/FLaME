#!/usr/bin/env python3
"""Live E2E testing campaign for FLAME FOMC task with BenchForge.

This script runs comprehensive tests with real API calls to validate:
1. Basic inference with default settings
2. Different prompt formats (zero_shot, few_shot, chain_of_thought)
3. Various batch sizes and sample counts
4. Evaluation metrics computation
5. Error handling and recovery
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIGS = {
    "test_1_basic": {
        "description": "Basic test with minimal samples",
        "num_samples": 3,
        "batch_size": 2,
        "max_tokens": 10,
        "temperature": 0.0,
        "prompt_format": "zero_shot",
    },
    "test_2_few_shot": {
        "description": "Test few-shot prompting",
        "num_samples": 5,
        "batch_size": 3,
        "max_tokens": 10,
        "temperature": 0.0,
        "prompt_format": "few_shot",
    },
    "test_3_chain_of_thought": {
        "description": "Test chain-of-thought prompting",
        "num_samples": 5,
        "batch_size": 2,
        "max_tokens": 50,
        "temperature": 0.0,
        "prompt_format": "chain_of_thought",
    },
    "test_4_larger_batch": {
        "description": "Test with larger batch size",
        "num_samples": 10,
        "batch_size": 5,
        "max_tokens": 10,
        "temperature": 0.0,
        "prompt_format": "zero_shot",
    },
    "test_5_temperature": {
        "description": "Test with non-zero temperature",
        "num_samples": 5,
        "batch_size": 3,
        "max_tokens": 10,
        "temperature": 0.3,
        "prompt_format": "zero_shot",
    },
}


class FOMLiveE2ETester:
    """Comprehensive E2E tester for FLAME FOMC task."""

    def __init__(
        self, model: str = "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
    ):
        """Initialize tester with model configuration."""
        self.model = model
        self.results_dir = Path("scratch/live_test_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.test_results = {}

    def setup_environment(self) -> bool:
        """Check and setup environment."""
        logger.info("\n" + "=" * 60)
        logger.info("Environment Setup")
        logger.info("=" * 60)

        # Check for API keys
        from dotenv import load_dotenv

        load_dotenv()

        api_keys_found = True
        # Check both possible key names for Together API
        together_key = os.getenv("TOGETHER_API_KEY") or os.getenv("TOGETHERAI_API_KEY")
        hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

        if together_key:
            logger.info("✓ Together API key is set")
            # Set it with the expected name for litellm
            os.environ["TOGETHER_API_KEY"] = together_key
        else:
            logger.warning("✗ Together API key is not set")
            api_keys_found = False

        if hf_key:
            logger.info("✓ HuggingFace token is set")
        else:
            logger.warning("✗ HuggingFace token is not set")
            api_keys_found = False

        if not api_keys_found:
            logger.error("Missing required API keys. Please set them in .env file.")
            return False

        # Register FLAME tasks
        try:
            from flame.tasks import register_all_flame_tasks

            registered = register_all_flame_tasks()
            logger.info(f"✓ Registered tasks: {registered}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to register tasks: {e}")
            return False

    def run_inference_test(
        self, test_name: str, config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Run a single inference test configuration."""
        logger.info("\n" + "-" * 60)
        logger.info(f"Running Test: {test_name}")
        logger.info(f"Description: {config['description']}")
        logger.info("-" * 60)

        try:
            from flame.benchforge import (
                FLAMEConfig,
                create_inference_engine,
                LLMConfig,
                LLMClient,
            )
            from flame.tasks import create_task

            # Create FOMC task with test configuration
            task_config = FLAMEConfig(
                name="fomc",
                dataset="fomc",
                huggingface_dataset="gtfintechlab/fomc_communication",
                num_samples=config["num_samples"],
                batch_size=config["batch_size"],
                prompt_format=config.get("prompt_format", "zero_shot"),
            )

            create_task("fomc", task_config)
            logger.info(f"✓ Created FOMC task with {config['num_samples']} samples")

            # Create LLM configuration
            llm_config = LLMConfig(
                provider="litellm",
                model=self.model,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
                top_p=1.0,
            )

            # Create LLM client
            llm_client = LLMClient(llm_config)
            logger.info(f"✓ Created LLM client for {self.model}")

            # Create inference engine
            output_dir = self.results_dir / test_name
            output_dir.mkdir(parents=True, exist_ok=True)

            engine = create_inference_engine(
                llm_client=llm_client, output_dir=output_dir
            )
            logger.info("✓ Created inference engine")

            # Run inference
            logger.info(f"Running inference with {config['prompt_format']} prompts...")
            start_time = time.time()

            result = engine.run(task="fomc", config=task_config)

            execution_time = time.time() - start_time

            # Analyze results
            df = result.results_df
            extraction_rate = (
                df["extracted_response"].notna().mean()
                if "extracted_response" in df.columns
                else 0
            )

            test_result = {
                "test_name": test_name,
                "config": config,
                "execution_time": execution_time,
                "samples_processed": len(df),
                "extraction_rate": extraction_rate,
                "output_path": str(result.output_path),
                "success": True,
                "error": None,
            }

            # Show sample predictions
            logger.info(f"✓ Inference completed in {execution_time:.2f}s")
            logger.info(f"  - Samples processed: {len(df)}")
            logger.info(f"  - Extraction rate: {extraction_rate:.1%}")

            if extraction_rate > 0:
                logger.info("  - Sample predictions:")
                for idx, row in df.head(3).iterrows():
                    pred = row.get("extracted_response", "N/A")
                    gt = row.get("ground_truth", "N/A")
                    logger.info(f"    {idx}: {pred} (GT: {gt})")

            # Quick accuracy check
            if "ground_truth" in df.columns and "extracted_response" in df.columns:
                correct = (df["extracted_response"] == df["ground_truth"]).sum()
                accuracy = correct / len(df)
                test_result["accuracy"] = accuracy
                logger.info(f"  - Quick accuracy: {accuracy:.1%} ({correct}/{len(df)})")

            return test_result

        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            import traceback

            traceback.print_exc()

            return {
                "test_name": test_name,
                "config": config,
                "success": False,
                "error": str(e),
            }

    def run_evaluation_test(
        self, test_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Run evaluation on inference results."""
        if not test_result.get("success") or not test_result.get("output_path"):
            logger.warning(
                f"Skipping evaluation for failed test: {test_result['test_name']}"
            )
            return None

        logger.info(f"\nEvaluating results from: {test_result['test_name']}")

        try:
            from flame.benchforge import create_evaluation_engine

            # Create evaluation engine
            eval_output_dir = self.results_dir / f"{test_result['test_name']}_eval"
            eval_output_dir.mkdir(parents=True, exist_ok=True)

            eval_engine = create_evaluation_engine(output_dir=eval_output_dir)

            # Run evaluation
            eval_result = eval_engine.evaluate(
                results_path=test_result["output_path"],
                task="fomc",
                metrics=["accuracy", "f1_macro"],
                save_results=True,
            )

            eval_data = {
                "test_name": test_result["test_name"],
                "num_samples": eval_result.num_samples,
                "metrics": eval_result.metrics,
                "num_errors": eval_result.num_errors,
                "success": True,
            }

            logger.info("✓ Evaluation completed:")
            for metric, value in eval_result.metrics.items():
                if isinstance(value, float):
                    logger.info(f"  - {metric}: {value:.4f}")

            return eval_data

        except Exception as e:
            logger.error(f"✗ Evaluation failed: {e}")
            return {
                "test_name": test_result["test_name"],
                "success": False,
                "error": str(e),
            }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test configurations."""
        logger.info("\n" + "=" * 60)
        logger.info("FLAME FOMC Live E2E Testing Campaign")
        logger.info("=" * 60)

        # Setup environment
        if not self.setup_environment():
            logger.error("Environment setup failed. Cannot proceed.")
            return {"success": False, "error": "Environment setup failed"}

        # Run each test configuration
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "tests": [],
            "evaluations": [],
            "summary": {},
        }

        for test_name, config in TEST_CONFIGS.items():
            # Run inference test
            test_result = self.run_inference_test(test_name, config)
            if test_result:
                all_results["tests"].append(test_result)

                # Run evaluation if inference succeeded
                if test_result.get("success"):
                    eval_result = self.run_evaluation_test(test_result)
                    if eval_result:
                        all_results["evaluations"].append(eval_result)

            # Small delay between tests to avoid rate limiting
            time.sleep(2)

        # Generate summary
        successful_tests = sum(1 for t in all_results["tests"] if t.get("success"))
        total_tests = len(all_results["tests"])

        all_results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "average_execution_time": sum(
                t.get("execution_time", 0)
                for t in all_results["tests"]
                if t.get("success")
            )
            / successful_tests
            if successful_tests > 0
            else 0,
            "average_extraction_rate": sum(
                t.get("extraction_rate", 0)
                for t in all_results["tests"]
                if t.get("success")
            )
            / successful_tests
            if successful_tests > 0
            else 0,
        }

        # Save all results
        results_path = (
            self.results_dir
            / f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"\n✓ Saved all results to: {results_path}")

        return all_results

    def print_summary(self, results: Dict[str, Any]):
        """Print test summary."""
        logger.info("\n" + "=" * 60)
        logger.info("Test Campaign Summary")
        logger.info("=" * 60)

        summary = results.get("summary", {})
        logger.info(f"Total Tests: {summary.get('total_tests', 0)}")
        logger.info(f"Successful: {summary.get('successful_tests', 0)}")
        logger.info(f"Success Rate: {summary.get('success_rate', 0):.1%}")
        logger.info(
            f"Avg Execution Time: {summary.get('average_execution_time', 0):.2f}s"
        )
        logger.info(
            f"Avg Extraction Rate: {summary.get('average_extraction_rate', 0):.1%}"
        )

        # Individual test results
        logger.info("\nIndividual Test Results:")
        for test in results.get("tests", []):
            status = "✓" if test.get("success") else "✗"
            logger.info(
                f"{status} {test['test_name']}: {test['config']['description']}"
            )
            if test.get("success"):
                logger.info(f"  - Execution: {test.get('execution_time', 0):.2f}s")
                logger.info(f"  - Extraction: {test.get('extraction_rate', 0):.1%}")
                if "accuracy" in test:
                    logger.info(f"  - Accuracy: {test['accuracy']:.1%}")
            else:
                logger.info(f"  - Error: {test.get('error', 'Unknown')}")

        # Evaluation results
        if results.get("evaluations"):
            logger.info("\nEvaluation Results:")
            for eval_result in results["evaluations"]:
                if eval_result.get("success"):
                    logger.info(f"✓ {eval_result['test_name']}:")
                    for metric, value in eval_result.get("metrics", {}).items():
                        if isinstance(value, float):
                            logger.info(f"  - {metric}: {value:.4f}")


def main():
    """Run the live E2E testing campaign."""
    # Allow model override from command line
    model = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
    )

    tester = FOMLiveE2ETester(model=model)
    results = tester.run_all_tests()
    tester.print_summary(results)

    # Return success/failure exit code
    success_rate = results.get("summary", {}).get("success_rate", 0)
    return 0 if success_rate >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
