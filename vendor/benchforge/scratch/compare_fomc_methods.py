#!/usr/bin/env python3
"""Unified test runner for comparing FOMC native FLAME vs BenchForge methods.

This script runs FOMC classification on multiple models using both methods
and compares their outputs to ensure BenchForge is a superset of native FLAME.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from dotenv import load_dotenv

# Setup paths for both FLAME and BenchForge
flame_root = Path(__file__).parent.parent
benchforge_root = Path(__file__).parent
sys.path.insert(0, str(flame_root))
sys.path.insert(0, str(benchforge_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("fomc_comparison.log")],
)
logger = logging.getLogger(__name__)

load_dotenv()

# Define the 5 diverse models to test
TEST_MODELS = [
    "together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo",  # Small/Fast
    "together_ai/mistralai/Mistral-7B-Instruct-v0.3",  # Small-Medium
    "together_ai/meta-llama/Llama-3.1-8B-Instruct-Turbo",  # Medium
    "together_ai/mistralai/Mistral-Small-24B-Instruct-2501",  # Medium-Large (if available)
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",  # Large/High-Quality
]

# Fallback models if some are not available
FALLBACK_MODELS = [
    "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",  # Known working model
    "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
    "together_ai/google/gemma-2b-it",
]


class FOMCComparison:
    """Compare FOMC results between native FLAME and BenchForge methods."""

    def __init__(self, num_samples: int = 10, batch_size: int = 5):
        """Initialize comparison runner.

        Args:
            num_samples: Number of samples to test (default 10 for quick test)
            batch_size: Batch size for processing
        """
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.results = {}
        self.comparison_metrics = {}

        # Check API key
        if not os.getenv("TOGETHERAI_API_KEY"):
            raise ValueError("TOGETHERAI_API_KEY not found in environment")

        logger.info(
            f"Initialized comparison with {num_samples} samples, batch size {batch_size}"
        )

    def run_native_flame(self, model: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run FOMC using native FLAME implementation.

        Args:
            model: Model identifier

        Returns:
            Tuple of (results_df, metrics_df)
        """
        logger.info(f"Running native FLAME with model: {model}")

        try:
            # Import FLAME modules
            from flame.code.fomc.fomc_inference import fomc_inference
            from flame.code.fomc.fomc_evaluate import fomc_evaluate
            from flame.utils.dataset_utils import safe_load_dataset

            # Create args object for FLAME
            class Args:
                pass

            args = Args()
            args.model = model
            args.batch_size = self.batch_size
            args.max_tokens = 128
            args.temperature = 0.0
            args.top_p = 0.9
            args.top_k = None
            args.repetition_penalty = 1.0
            args.prompt_format = "zero_shot"
            args.dataset = "fomc"
            args.task = "fomc"

            # Load dataset
            dataset = safe_load_dataset(
                "gtfintechlab/fomc_communication", trust_remote_code=True
            )
            test_data = dataset["test"]

            # Limit to num_samples for testing
            if self.num_samples and self.num_samples < len(test_data):
                # Create a subset for testing
                import tempfile

                subset_data = [test_data[i] for i in range(self.num_samples)]

                # Mock the dataset loading to return our subset
                import flame.code.fomc.fomc_inference as fomc_module

                original_load = fomc_module.load_fomc_dataset

                def mock_load():
                    return subset_data

                fomc_module.load_fomc_dataset = mock_load

            # Run inference
            start_time = time.time()
            results_df = fomc_inference(args)
            inference_time = time.time() - start_time

            # Run evaluation
            # Save results to temp file for evaluation
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                results_df.to_csv(f.name, index=False)
                temp_path = f.name

            eval_results_df, metrics_df = fomc_evaluate(temp_path, args)

            # Clean up temp file
            Path(temp_path).unlink()

            # Restore original function if mocked
            if self.num_samples:
                fomc_module.load_fomc_dataset = original_load

            logger.info(f"Native FLAME completed in {inference_time:.2f}s")

            return eval_results_df, metrics_df

        except Exception as e:
            logger.error(f"Native FLAME failed: {e}")
            raise

    def run_benchforge(self, model: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run FOMC using BenchForge implementation.

        Args:
            model: Model identifier

        Returns:
            Tuple of (results_df, metrics_df)
        """
        logger.info(f"Running BenchForge with model: {model}")

        try:
            # Import BenchForge modules
            from bench_forge.flame.tasks.fomc import FOMCConfig, FOMCTask
            from bench_forge.llm.client import LLMClient
            from bench_forge.llm.config import LLMConfig
            from bench_forge.tasks.config import PromptFormat

            # Create configuration
            config = FOMCConfig(
                name="fomc",
                dataset="fomc_communication",
                huggingface_dataset="gtfintechlab/fomc_communication",
                dataset_split="test",
                prompt_format=PromptFormat.ZERO_SHOT,
                batch_size=self.batch_size,
                max_tokens=128,
                temperature=0.0,
                top_p=0.9,
                seed=42,
                metrics=["accuracy", "f1", "precision", "recall"],
                model=model,
            )

            # Initialize task
            task = FOMCTask(config)

            # Initialize LLM client
            llm_config = LLMConfig(
                provider="litellm",
                model=model,
                max_tokens=128,
                temperature=0.0,
                top_p=0.9,
                seed=42,
                api_key=os.getenv("TOGETHERAI_API_KEY"),
            )
            llm_client = LLMClient(llm_config)

            # Set LLM client for Strategy 7 (optional LLM-based extraction)
            task.set_llm_client(llm_client)

            # Load dataset
            dataset = task.load_dataset("test")

            # Limit samples for testing
            if self.num_samples and self.num_samples < len(dataset):
                samples = [dataset[i] for i in range(self.num_samples)]
            else:
                samples = dataset

            logger.info(f"Processing {len(samples)} samples")

            # Generate prompts
            prompts = task.process_batch(samples, config.prompt_format)

            # Process through LLM
            start_time = time.time()
            responses = []

            for i in range(0, len(prompts), self.batch_size):
                batch = prompts[i : i + self.batch_size]
                batch_responses = llm_client.complete_batch(batch)
                responses.extend(batch_responses)
                logger.debug(f"Processed batch {i // self.batch_size + 1}")

            inference_time = time.time() - start_time

            # Extract labels
            extracted_labels = []
            for response in responses:
                if hasattr(response, "choices"):
                    text = (
                        response.choices[0].message.content if response.choices else ""
                    )
                else:
                    text = str(response) if response else ""
                label = task.extract_label_from_response(text)
                extracted_labels.append(label)

            # Format results with evaluation
            results_dict = task.format_results_with_evaluation(
                samples, prompts, responses, extracted_labels
            )

            results_df = results_dict["results"]
            metrics_df = results_dict["metrics"]

            logger.info(f"BenchForge completed in {inference_time:.2f}s")

            return results_df, metrics_df

        except Exception as e:
            logger.error(f"BenchForge failed: {e}")
            raise

    def compare_results(
        self,
        flame_results: pd.DataFrame,
        flame_metrics: pd.DataFrame,
        benchforge_results: pd.DataFrame,
        benchforge_metrics: pd.DataFrame,
        model: str,
    ) -> Dict[str, Any]:
        """Compare results between native FLAME and BenchForge.

        Args:
            flame_results: Native FLAME results DataFrame
            flame_metrics: Native FLAME metrics DataFrame
            benchforge_results: BenchForge results DataFrame
            benchforge_metrics: BenchForge metrics DataFrame
            model: Model name for reporting

        Returns:
            Comparison dictionary
        """
        comparison = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(flame_results),
        }

        # Compare column names
        flame_cols = set(flame_results.columns)
        benchforge_cols = set(benchforge_results.columns)

        comparison["columns"] = {
            "flame_only": list(flame_cols - benchforge_cols),
            "benchforge_only": list(benchforge_cols - flame_cols),
            "common": list(flame_cols & benchforge_cols),
        }

        # Check if BenchForge is a superset
        comparison["is_superset"] = len(flame_cols - benchforge_cols) == 0

        # Compare metrics
        flame_accuracy = flame_metrics[flame_metrics["Metric"] == "Accuracy"][
            "Value"
        ].values[0]
        benchforge_accuracy = benchforge_metrics[
            benchforge_metrics["Metric"] == "Accuracy"
        ]["Value"].values[0]

        comparison["metrics"] = {
            "flame_accuracy": float(flame_accuracy),
            "benchforge_accuracy": float(benchforge_accuracy),
            "accuracy_diff": abs(float(flame_accuracy) - float(benchforge_accuracy)),
        }

        # Compare extraction success
        flame_extracted = (flame_results["extracted_labels"] != -1).sum()

        # BenchForge might have text labels, convert for comparison
        if "extracted_labels_numeric" in benchforge_results.columns:
            benchforge_extracted = (
                benchforge_results["extracted_labels_numeric"] != -1
            ).sum()
        else:
            benchforge_extracted = benchforge_results["extracted_labels"].notna().sum()

        comparison["extraction"] = {
            "flame_success": flame_extracted,
            "benchforge_success": benchforge_extracted,
            "flame_rate": flame_extracted / len(flame_results),
            "benchforge_rate": benchforge_extracted / len(benchforge_results),
        }

        # Log comparison
        logger.info(f"Comparison for {model}:")
        logger.info(f"  Columns - FLAME only: {comparison['columns']['flame_only']}")
        logger.info(
            f"  Columns - BenchForge only: {comparison['columns']['benchforge_only']}"
        )
        logger.info(f"  Is BenchForge superset: {comparison['is_superset']}")
        logger.info(
            f"  Accuracy - FLAME: {comparison['metrics']['flame_accuracy']:.3f}"
        )
        logger.info(
            f"  Accuracy - BenchForge: {comparison['metrics']['benchforge_accuracy']:.3f}"
        )
        logger.info(
            f"  Extraction - FLAME: {comparison['extraction']['flame_rate']:.1%}"
        )
        logger.info(
            f"  Extraction - BenchForge: {comparison['extraction']['benchforge_rate']:.1%}"
        )

        return comparison

    def run_comparison(self, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run full comparison across multiple models.

        Args:
            models: List of model identifiers to test

        Returns:
            Full comparison results
        """
        if models is None:
            models = TEST_MODELS[:3]  # Default to first 3 for quick test

        all_comparisons = []

        for model in models:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Testing model: {model}")
            logger.info(f"{'=' * 60}")

            try:
                # Run native FLAME
                flame_results, flame_metrics = self.run_native_flame(model)

                # Run BenchForge
                benchforge_results, benchforge_metrics = self.run_benchforge(model)

                # Compare results
                comparison = self.compare_results(
                    flame_results,
                    flame_metrics,
                    benchforge_results,
                    benchforge_metrics,
                    model,
                )

                all_comparisons.append(comparison)

                # Save individual results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path("comparison_results") / timestamp
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save DataFrames
                model_name = model.split("/")[-1]
                flame_results.to_csv(
                    output_dir / f"flame_{model_name}.csv", index=False
                )
                benchforge_results.to_csv(
                    output_dir / f"benchforge_{model_name}.csv", index=False
                )

                # Save metrics
                flame_metrics.to_csv(
                    output_dir / f"flame_metrics_{model_name}.csv", index=False
                )
                benchforge_metrics.to_csv(
                    output_dir / f"benchforge_metrics_{model_name}.csv", index=False
                )

            except Exception as e:
                logger.error(f"Failed to test model {model}: {e}")
                all_comparisons.append(
                    {
                        "model": model,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        # Summary report
        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_models": len(models),
            "num_samples_per_model": self.num_samples,
            "comparisons": all_comparisons,
            "overall_superset": all(
                c.get("is_superset", False) for c in all_comparisons if "error" not in c
            ),
        }

        # Save summary
        output_path = Path("comparison_results") / "summary.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{'=' * 60}")
        logger.info("COMPARISON SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"Models tested: {len(models)}")
        logger.info(f"Samples per model: {self.num_samples}")
        logger.info(f"BenchForge is superset: {summary['overall_superset']}")

        return summary


def main():
    """Main entry point for comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare FOMC native FLAME vs BenchForge implementations"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to test (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for processing (default: 5)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="List of models to test (default: first 3 from TEST_MODELS)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick test with 1 model and 5 samples"
    )

    args = parser.parse_args()

    # Quick test mode
    if args.quick:
        args.num_samples = 5
        models = [TEST_MODELS[0]]  # Just the smallest model
    else:
        models = args.models or TEST_MODELS[:3]  # Default to first 3

    # Run comparison
    comparison = FOMCComparison(
        num_samples=args.num_samples, batch_size=args.batch_size
    )

    results = comparison.run_comparison(models)

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for comp in results["comparisons"]:
        if "error" in comp:
            print(f"❌ {comp['model']}: ERROR - {comp['error']}")
        else:
            status = "✅" if comp["is_superset"] else "⚠️"
            print(f"{status} {comp['model']}:")
            print(f"   - BenchForge is superset: {comp['is_superset']}")
            print(f"   - Accuracy diff: {comp['metrics']['accuracy_diff']:.3f}")
            print(
                f"   - Extraction rates: FLAME={comp['extraction']['flame_rate']:.1%}, BenchForge={comp['extraction']['benchforge_rate']:.1%}"
            )

    print(
        f"\nOverall: BenchForge is {'✅ SUPERSET' if results['overall_superset'] else '⚠️ NOT SUPERSET'} of native FLAME"
    )
    print("Results saved to: comparison_results/")

    return 0 if results["overall_superset"] else 1


if __name__ == "__main__":
    sys.exit(main())
