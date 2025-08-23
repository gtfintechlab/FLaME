#!/usr/bin/env python3
"""Compare FinEntity implementations: BenchForge vs Native FLAME.

This script:
1. Runs FinEntity with BenchForge implementation (entity+sentiment extraction)
2. Runs FinEntity with native FLAME implementation (entity+sentiment extraction)
3. Compares results and metrics on the full dataset
4. Tests with live TogetherAI API calls
5. Generates detailed comparison report
"""

import os
import sys
import time
import json
import logging
import pandas as pd
import argparse
from datetime import datetime
from typing import Dict, List
from litellm import completion
from pathlib import Path

# Add paths for both implementations
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_finentity_dataset():
    """Load the complete FinEntity dataset."""
    print("Loading FinEntity dataset...")

    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "gtfintechlab/finentity", name="5768", trust_remote_code=True
        )
        test_data = dataset["test"]

        # Convert to list of dictionaries
        samples = []
        for item in test_data:
            samples.append(
                {"content": item["content"], "annotations": item["annotations"]}
            )

        print(f"Loaded {len(samples)} samples from FinEntity dataset")
        return samples

    except Exception as e:
        logger.error(f"Failed to load FinEntity dataset: {e}")
        return []


def run_benchforge_finentity(
    samples: List[Dict],
    model: str = "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    batch_size: int = 1,
):
    """Run FinEntity using BenchForge implementation."""
    print("\n" + "=" * 70)
    print("BENCHFORGE FINENTITY IMPLEMENTATION")
    print("=" * 70)
    print(f"Batch size: {batch_size}")

    from bench_forge.flame.tasks.finentity import FinEntityTask, FinEntityConfig
    from bench_forge.tasks.config import PromptFormat

    # Configure task
    config = FinEntityConfig(
        name="finentity", prompt_format=PromptFormat.ZERO_SHOT, batch_size=batch_size
    )
    task = FinEntityTask(config)

    results = []
    total_api_time = 0
    successful_extractions = 0

    if batch_size == 1:
        # Individual processing for detailed tracking
        for i, sample in enumerate(samples, 1):
            print(f"\n--- Sample {i}/{len(samples)} ---")
            print(f"Content: {sample['content'][:100]}...")
            print(f"Ground truth: {sample.get('annotations', 'N/A')}")

            # Create prompt
            prompt = task.create_prompt(sample)

            try:
                # Make API call
                start_time = time.time()
                response = completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=512,  # Sufficient for entity extraction
                )
                api_time = time.time() - start_time
                total_api_time += api_time

                # Extract response
                response_text = response.choices[0].message.content
                print(f"Raw response: {response_text[:200]}...")

                # Extract entities using task's extraction logic
                extracted = task.extract_response(response_text, sample)
                print(f"Extracted entities: {extracted}")
                print(f"API time: {api_time:.2f}s")

                if extracted:
                    successful_extractions += 1

                results.append(
                    {
                        "implementation": "BenchForge",
                        "sample_id": i,
                        "content": sample["content"],
                        "ground_truth": sample.get("annotations"),
                        "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                        "raw_response": response_text,
                        "extracted_entities": extracted,
                        "api_time": api_time,
                        "success": bool(extracted),
                    }
                )

            except Exception as e:
                logger.error(f"BenchForge error on sample {i}: {e}")
                results.append(
                    {
                        "implementation": "BenchForge",
                        "sample_id": i,
                        "content": sample["content"],
                        "ground_truth": sample.get("annotations"),
                        "prompt": "",
                        "raw_response": str(e),
                        "extracted_entities": [],
                        "api_time": 0,
                        "success": False,
                    }
                )
    else:
        # Batch processing using BenchForge's batch capabilities
        print(f"Processing {len(samples)} samples in batches of {batch_size}")

        # Use BenchForge's batch processing
        prompts = task.process_batch(samples)

        # Process batches
        from flame.utils.batch_utils import chunk_list

        sample_batches = chunk_list(samples, batch_size)
        prompt_batches = chunk_list(prompts, batch_size)

        for batch_idx, (sample_batch, prompt_batch) in enumerate(
            zip(sample_batches, prompt_batches)
        ):
            print(
                f"\n--- Batch {batch_idx + 1}/{len(sample_batches)} ({len(sample_batch)} samples) ---"
            )

            # Create messages for batch
            messages_batch = [
                [{"role": "user", "content": prompt}] for prompt in prompt_batch
            ]

            try:
                start_time = time.time()
                batch_responses = []

                # Process each message in the batch (TogetherAI doesn't support true batch API)
                for msg in messages_batch:
                    response = completion(
                        model=model, messages=msg, temperature=0, max_tokens=512
                    )
                    batch_responses.append(response)

                batch_time = time.time() - start_time
                total_api_time += batch_time

                print(
                    f"Batch API time: {batch_time:.2f}s ({batch_time / len(sample_batch):.2f}s per sample)"
                )

                # Process responses
                for i, (sample, prompt, response) in enumerate(
                    zip(sample_batch, prompt_batch, batch_responses)
                ):
                    sample_id = batch_idx * batch_size + i + 1

                    try:
                        response_text = response.choices[0].message.content
                        extracted = task.extract_response(response_text, sample)

                        if extracted:
                            successful_extractions += 1

                        results.append(
                            {
                                "implementation": "BenchForge",
                                "sample_id": sample_id,
                                "content": sample["content"],
                                "ground_truth": sample.get("annotations"),
                                "prompt": prompt[:500] + "..."
                                if len(prompt) > 500
                                else prompt,
                                "raw_response": response_text,
                                "extracted_entities": extracted,
                                "api_time": batch_time
                                / len(sample_batch),  # Approximate per-sample time
                                "success": bool(extracted),
                            }
                        )

                    except Exception as e:
                        logger.error(f"Error processing sample {sample_id}: {e}")
                        results.append(
                            {
                                "implementation": "BenchForge",
                                "sample_id": sample_id,
                                "content": sample["content"],
                                "ground_truth": sample.get("annotations"),
                                "prompt": "",
                                "raw_response": str(e),
                                "extracted_entities": [],
                                "api_time": 0,
                                "success": False,
                            }
                        )

            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {e}")
                # Add failed results for all samples in batch
                for i, sample in enumerate(sample_batch):
                    sample_id = batch_idx * batch_size + i + 1
                    results.append(
                        {
                            "implementation": "BenchForge",
                            "sample_id": sample_id,
                            "content": sample["content"],
                            "ground_truth": sample.get("annotations"),
                            "prompt": "",
                            "raw_response": str(e),
                            "extracted_entities": [],
                            "api_time": 0,
                            "success": False,
                        }
                    )

    success_rate = (successful_extractions / len(samples)) * 100 if samples else 0
    avg_time = total_api_time / len(samples) if samples else 0

    print("\nBenchForge Summary:")
    print(
        f"Success rate: {successful_extractions}/{len(samples)} ({success_rate:.1f}%)"
    )
    print(f"Total API time: {total_api_time:.2f}s")
    print(f"Average time per sample: {avg_time:.2f}s")

    return results


def run_flame_finentity(
    samples: List[Dict],
    model: str = "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    batch_size: int = 1,
):
    """Run FinEntity using native FLAME implementation."""
    print("\n" + "=" * 70)
    print("NATIVE FLAME FINENTITY IMPLEMENTATION")
    print("=" * 70)
    print(f"Batch size: {batch_size}")

    try:
        # Import FLAME modules - try different import paths
        try:
            from flame.code.prompts import PromptFormat as FLAMEPromptFormat, get_prompt
            from flame.code.finentity.finentity_evaluate import (
                parse_json_content,
                sanitize_json_string,
            )
        except ImportError:
            # Try alternative import path
            sys.path.insert(
                0,
                os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..", "src")
                ),
            )
            from flame.code.prompts import PromptFormat as FLAMEPromptFormat, get_prompt
            from flame.code.finentity.finentity_evaluate import (
                parse_json_content,
                sanitize_json_string,
            )
        import ast  # noqa: F401

        # Get FLAME prompt
        finentity_prompt = get_prompt("finentity", FLAMEPromptFormat.ZERO_SHOT)
        if finentity_prompt is None:
            # Try to import and register prompts
            try:
                import flame.code.prompts.zeroshot  # noqa: F401
            except ImportError:
                # Import with src path
                import src.flame.code.prompts.zeroshot  # noqa: F401
            finentity_prompt = get_prompt("finentity", FLAMEPromptFormat.ZERO_SHOT)

        if finentity_prompt is None:
            logger.error("Could not load FLAME FinEntity prompt")
            return []

        results = []
        total_api_time = 0
        successful_extractions = 0

        for i, sample in enumerate(samples, 1):
            print(f"\n--- Sample {i}/{len(samples)} ---")
            print(f"Content: {sample['content'][:100]}...")
            print(f"Ground truth: {sample.get('annotations', 'N/A')}")

            # Create prompt using FLAME format
            prompt = finentity_prompt(sample["content"])

            try:
                # Make API call
                start_time = time.time()
                response = completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=512,
                )
                api_time = time.time() - start_time
                total_api_time += api_time

                # Extract response
                response_text = response.choices[0].message.content
                print(f"Raw response: {response_text[:200]}...")

                # Extract entities using FLAME logic
                extracted = []
                try:
                    # Clean and parse response
                    sanitized = sanitize_json_string(response_text)
                    parsed = parse_json_content(sanitized)
                    if isinstance(parsed, list):
                        extracted = parsed
                except Exception as e:
                    logger.debug(f"Failed to parse response: {e}")

                print(f"Extracted entities: {extracted}")
                print(f"API time: {api_time:.2f}s")

                if extracted:
                    successful_extractions += 1

                results.append(
                    {
                        "implementation": "FLAME",
                        "sample_id": i,
                        "content": sample["content"],
                        "ground_truth": sample.get("annotations"),
                        "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                        "raw_response": response_text,
                        "extracted_entities": extracted,
                        "api_time": api_time,
                        "success": bool(extracted),
                    }
                )

            except Exception as e:
                logger.error(f"FLAME API error on sample {i}: {e}")
                results.append(
                    {
                        "implementation": "FLAME",
                        "sample_id": i,
                        "content": sample["content"],
                        "ground_truth": sample.get("annotations"),
                        "prompt": "",
                        "raw_response": str(e),
                        "extracted_entities": [],
                        "api_time": 0,
                        "success": False,
                    }
                )

        success_rate = (successful_extractions / len(samples)) * 100 if samples else 0
        avg_time = total_api_time / len(samples) if samples else 0

        print("\nFLAME Summary:")
        print(
            f"Success rate: {successful_extractions}/{len(samples)} ({success_rate:.1f}%)"
        )
        print(f"Total API time: {total_api_time:.2f}s")
        print(f"Average time per sample: {avg_time:.2f}s")

        return results

    except ImportError as e:
        logger.error(f"Could not import FLAME modules: {e}")
        return []


def calculate_entity_metrics(
    extracted_entities: List[Dict], ground_truth_entities: List[Dict]
):
    """Calculate metrics for entity extraction."""
    if isinstance(ground_truth_entities, str):
        try:
            ground_truth_entities = json.loads(ground_truth_entities)
        except json.JSONDecodeError:
            ground_truth_entities = []

    if not isinstance(ground_truth_entities, list):
        ground_truth_entities = []

    if not isinstance(extracted_entities, list):
        extracted_entities = []

    # Normalize for comparison
    def normalize_entity(entity):
        if not isinstance(entity, dict):
            return None
        return {
            "value": entity.get("value", "").strip().lower(),
            "tag": entity.get("tag", "").strip().lower(),
            "label": entity.get("label", "").strip().lower(),
        }

    norm_extracted = [
        normalize_entity(e) for e in extracted_entities if normalize_entity(e)
    ]
    norm_ground_truth = [
        normalize_entity(e) for e in ground_truth_entities if normalize_entity(e)
    ]

    # Calculate matches
    matches = 0
    for entity in norm_extracted:
        if entity in norm_ground_truth:
            matches += 1

    # Calculate metrics
    precision = matches / len(norm_extracted) if norm_extracted else 0
    recall = matches / len(norm_ground_truth) if norm_ground_truth else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
        "extracted_count": len(norm_extracted),
        "ground_truth_count": len(norm_ground_truth),
    }


def compare_results(benchforge_results: List[Dict], flame_results: List[Dict]):
    """Compare results from both implementations."""
    print("\n" + "=" * 70)
    print("DETAILED COMPARISON RESULTS")
    print("=" * 70)

    if not benchforge_results or not flame_results:
        print("⚠️ Missing results from one or both implementations")
        return {}

    # Aggregate metrics
    bf_metrics = []
    flame_metrics = []
    agreement_count = 0

    # Compare sample by sample
    for i, (bf_result, flame_result) in enumerate(
        zip(benchforge_results, flame_results)
    ):
        print(f"\n--- Sample {i + 1} Comparison ---")

        # Calculate individual metrics
        bf_sample_metrics = calculate_entity_metrics(
            bf_result["extracted_entities"], bf_result["ground_truth"]
        )
        flame_sample_metrics = calculate_entity_metrics(
            flame_result["extracted_entities"], flame_result["ground_truth"]
        )

        bf_metrics.append(bf_sample_metrics)
        flame_metrics.append(flame_sample_metrics)

        # Check agreement between implementations
        if bf_result["extracted_entities"] == flame_result["extracted_entities"] or str(
            bf_result["extracted_entities"]
        ) == str(flame_result["extracted_entities"]):
            agreement_count += 1
            print("✓ Implementations AGREE")
        else:
            print("✗ Implementations DIFFER")
            print(f"  BenchForge: {bf_result['extracted_entities']}")
            print(f"  FLAME: {flame_result['extracted_entities']}")

    # Calculate aggregate metrics
    avg_bf_metrics = {
        "precision": sum(m["precision"] for m in bf_metrics) / len(bf_metrics),
        "recall": sum(m["recall"] for m in bf_metrics) / len(bf_metrics),
        "f1": sum(m["f1"] for m in bf_metrics) / len(bf_metrics),
    }

    avg_flame_metrics = {
        "precision": sum(m["precision"] for m in flame_metrics) / len(flame_metrics),
        "recall": sum(m["recall"] for m in flame_metrics) / len(flame_metrics),
        "f1": sum(m["f1"] for m in flame_metrics) / len(flame_metrics),
    }

    # Performance metrics
    bf_success_rate = sum(1 for r in benchforge_results if r["success"]) / len(
        benchforge_results
    )
    flame_success_rate = sum(1 for r in flame_results if r["success"]) / len(
        flame_results
    )

    bf_avg_time = sum(r["api_time"] for r in benchforge_results) / len(
        benchforge_results
    )
    flame_avg_time = sum(r["api_time"] for r in flame_results) / len(flame_results)

    agreement_rate = agreement_count / len(benchforge_results)

    print("\n" + "=" * 50)
    print("AGGREGATE METRICS COMPARISON")
    print("=" * 50)

    print("\nBenchForge Metrics:")
    print(f"  Precision: {avg_bf_metrics['precision']:.4f}")
    print(f"  Recall: {avg_bf_metrics['recall']:.4f}")
    print(f"  F1 Score: {avg_bf_metrics['f1']:.4f}")
    print(f"  Success Rate: {bf_success_rate:.4f}")
    print(f"  Avg Time: {bf_avg_time:.2f}s")

    print("\nFLAME Metrics:")
    print(f"  Precision: {avg_flame_metrics['precision']:.4f}")
    print(f"  Recall: {avg_flame_metrics['recall']:.4f}")
    print(f"  F1 Score: {avg_flame_metrics['f1']:.4f}")
    print(f"  Success Rate: {flame_success_rate:.4f}")
    print(f"  Avg Time: {flame_avg_time:.2f}s")

    print(
        f"\nImplementation Agreement: {agreement_count}/{len(benchforge_results)} ({agreement_rate:.1%})"
    )

    return {
        "benchforge": avg_bf_metrics,
        "flame": avg_flame_metrics,
        "agreement_rate": agreement_rate,
        "benchforge_success_rate": bf_success_rate,
        "flame_success_rate": flame_success_rate,
        "benchforge_avg_time": bf_avg_time,
        "flame_avg_time": flame_avg_time,
    }


def save_results(
    benchforge_results: List[Dict], flame_results: List[Dict], comparison_metrics: Dict
):
    """Save results to CSV files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    output_dir = Path("results/finentity_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save BenchForge results
    bf_df = pd.DataFrame(benchforge_results)
    bf_path = output_dir / f"benchforge_finentity_{timestamp}.csv"
    bf_df.to_csv(bf_path, index=False)
    print(f"\nSaved BenchForge results to: {bf_path}")

    # Save FLAME results
    flame_df = pd.DataFrame(flame_results)
    flame_path = output_dir / f"flame_finentity_{timestamp}.csv"
    flame_df.to_csv(flame_path, index=False)
    print(f"Saved FLAME results to: {flame_path}")

    # Save comparison metrics
    metrics_path = output_dir / f"comparison_metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(comparison_metrics, f, indent=2)
    print(f"Saved comparison metrics to: {metrics_path}")

    return str(bf_path), str(flame_path), str(metrics_path)


def main():
    parser = argparse.ArgumentParser(description="Compare FinEntity implementations")
    parser.add_argument(
        "--model",
        default="together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
        help="Model to use for comparison",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for both implementations (default: 1 for detailed tracking)",
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save results to CSV files"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FINENTITY IMPLEMENTATION COMPARISON")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Sample limit: {args.limit if args.limit else 'All samples'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load dataset
    samples = load_finentity_dataset()
    if not samples:
        print("❌ Failed to load dataset")
        return

    # Limit samples if requested
    if args.limit:
        samples = samples[: args.limit]
        print(f"Limited to first {len(samples)} samples")

    # Run both implementations
    benchforge_results = run_benchforge_finentity(samples, args.model, args.batch_size)
    flame_results = run_flame_finentity(samples, args.model, args.batch_size)

    # Compare results
    comparison_metrics = compare_results(benchforge_results, flame_results)

    # Save results
    if args.save and benchforge_results and flame_results:
        save_results(benchforge_results, flame_results, comparison_metrics)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)

    return comparison_metrics


if __name__ == "__main__":
    main()
