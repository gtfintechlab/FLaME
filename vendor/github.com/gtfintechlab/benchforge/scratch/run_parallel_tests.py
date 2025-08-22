#!/usr/bin/env python3
"""Orchestrate parallel execution of FOMC tests across multiple models and methods.

This script spawns parallel processes to test 5 models with both FLAME and BenchForge,
enabling efficient comparison with 50 samples each.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - ORCHESTRATOR - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/orchestrator.log")
    ]
)
logger = logging.getLogger(__name__)

# Define models to test
TEST_MODELS = [
    "together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "together_ai/mistralai/Mistral-7B-Instruct-v0.3",
    "together_ai/meta-llama/Llama-3.1-8B-Instruct-Turbo",
    "together_ai/mistralai/Mistral-Small-24B-Instruct-2501",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
]


def run_single_test(model: str, method: str, samples: int, batch_size: int) -> Dict:
    """Run a single test in a subprocess.
    
    Args:
        model: Model identifier
        method: 'flame' or 'benchforge'
        samples: Number of samples
        batch_size: Batch size
    
    Returns:
        Result dictionary
    """
    model_name = model.split("/")[-1].lower().replace("-instruct", "").replace("-turbo", "")
    
    logger.info(f"Starting {method} test for {model_name}")
    
    # Determine script to run
    if method == "flame":
        script = "benchforge/run_single_flame_test.py"
    else:
        script = "benchforge/run_single_benchforge_test.py"
    
    # Build command
    cmd = [
        "uv", "run", "python", script,
        "--model", model,
        "--samples", str(samples),
        "--batch-size", str(batch_size)
    ]
    
    # Run subprocess
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per test
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ {method} test for {model_name} completed in {elapsed:.1f}s")
            
            # Load summary file
            summary_file = Path(f"results/{method}/{model_name}_summary.json")
            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                summary["elapsed_time"] = elapsed
                return summary
            else:
                return {
                    "model": model,
                    "model_name": model_name,
                    "method": method,
                    "status": "success",
                    "elapsed_time": elapsed,
                    "warning": "Summary file not found"
                }
        else:
            logger.error(f"❌ {method} test for {model_name} failed")
            logger.error(f"STDERR: {result.stderr}")
            
            return {
                "model": model,
                "model_name": model_name,
                "method": method,
                "status": "failed",
                "error": result.stderr[-500:],  # Last 500 chars of error
                "elapsed_time": elapsed
            }
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏱️ {method} test for {model_name} timed out")
        return {
            "model": model,
            "model_name": model_name,
            "method": method,
            "status": "timeout",
            "elapsed_time": 600
        }
    except Exception as e:
        logger.error(f"💥 {method} test for {model_name} crashed: {e}")
        return {
            "model": model,
            "model_name": model_name,
            "method": method,
            "status": "error",
            "error": str(e),
            "elapsed_time": time.time() - start_time
        }


def run_parallel_tests(
    models: List[str],
    samples: int = 50,
    batch_size: int = 10,
    max_workers: int = 4
) -> Dict:
    """Run all tests in parallel.
    
    Args:
        models: List of model identifiers
        samples: Number of samples per test
        batch_size: Batch size for processing
        max_workers: Maximum parallel workers
    
    Returns:
        Results dictionary
    """
    logger.info("="*60)
    logger.info("Starting Parallel FOMC Testing")
    logger.info(f"Models: {len(models)}")
    logger.info(f"Samples: {samples}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max workers: {max_workers}")
    logger.info("="*60)
    
    # Create result directories
    Path("results/flame").mkdir(parents=True, exist_ok=True)
    Path("results/benchforge").mkdir(parents=True, exist_ok=True)
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    Path("results/comparison").mkdir(parents=True, exist_ok=True)
    
    # Create test tasks
    tasks = []
    for model in models:
        tasks.append((model, "flame", samples, batch_size))
        tasks.append((model, "benchforge", samples, batch_size))
    
    logger.info(f"Created {len(tasks)} test tasks")
    
    # Run tests in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_single_test, *task): task
            for task in tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                
                status_icon = "✅" if result.get("status") == "success" else "❌"
                logger.info(
                    f"{status_icon} Completed: {task[1]} - {task[0].split('/')[-1]} "
                    f"({len(results)}/{len(tasks)})"
                )
            except Exception as e:
                logger.error(f"Task {task} generated exception: {e}")
                results.append({
                    "model": task[0],
                    "method": task[1],
                    "status": "exception",
                    "error": str(e)
                })
    
    logger.info("="*60)
    logger.info(f"All tests completed. Results: {len(results)}/{len(tasks)}")
    
    # Analyze results
    summary = analyze_results(results, models)
    
    # Save complete results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path("results/comparison") / f"parallel_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "models": models,
            "samples": samples,
            "batch_size": batch_size,
            "results": results,
            "summary": summary
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    return summary


def analyze_results(results: List[Dict], models: List[str]) -> Dict:
    """Analyze test results and generate summary.
    
    Args:
        results: List of test results
        models: List of tested models
    
    Returns:
        Summary dictionary
    """
    logger.info("\nAnalyzing results...")
    
    # Group results by model
    model_results = {}
    for result in results:
        model_name = result.get("model_name", result["model"].split("/")[-1])
        if model_name not in model_results:
            model_results[model_name] = {}
        model_results[model_name][result["method"]] = result
    
    # Compare each model
    comparisons = []
    for model_name, methods in model_results.items():
        comparison = {
            "model": model_name,
            "flame": methods.get("flame", {}),
            "benchforge": methods.get("benchforge", {})
        }
        
        # Calculate comparison metrics if both succeeded
        if (methods.get("flame", {}).get("status") == "success" and 
            methods.get("benchforge", {}).get("status") == "success"):
            
            flame = methods["flame"]
            benchforge = methods["benchforge"]
            
            comparison["metrics"] = {
                "accuracy_diff": abs(flame.get("accuracy", 0) - benchforge.get("accuracy", 0)),
                "f1_diff": abs(flame.get("f1_score", 0) - benchforge.get("f1_score", 0)),
                "extraction_diff": abs(flame.get("extraction_rate", 0) - benchforge.get("extraction_rate", 0)),
                "time_diff": benchforge.get("inference_time", 0) - flame.get("inference_time", 0)
            }
            
            comparison["benchforge_better"] = (
                benchforge.get("accuracy", 0) >= flame.get("accuracy", 0) and
                benchforge.get("extraction_rate", 0) >= flame.get("extraction_rate", 0)
            )
        
        comparisons.append(comparison)
    
    # Overall summary
    successful_tests = sum(1 for r in results if r.get("status") == "success")
    failed_tests = sum(1 for r in results if r.get("status") != "success")
    
    summary = {
        "total_tests": len(results),
        "successful": successful_tests,
        "failed": failed_tests,
        "success_rate": successful_tests / len(results) if results else 0,
        "comparisons": comparisons,
        "benchforge_superset": all(
            c.get("benchforge_better", False) 
            for c in comparisons 
            if "metrics" in c
        )
    }
    
    # Log summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Total tests: {summary['total_tests']}")
    logger.info(f"Successful: {summary['successful']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Success rate: {summary['success_rate']:.1%}")
    logger.info(f"BenchForge is superset: {summary['benchforge_superset']}")
    
    for comp in comparisons:
        if "metrics" in comp:
            logger.info(f"\n{comp['model']}:")
            logger.info(f"  FLAME accuracy: {comp['flame'].get('accuracy', 0):.3f}")
            logger.info(f"  BenchForge accuracy: {comp['benchforge'].get('accuracy', 0):.3f}")
            logger.info(f"  Accuracy diff: {comp['metrics']['accuracy_diff']:.3f}")
            logger.info(f"  BenchForge better: {comp.get('benchforge_better', False)}")
    
    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run parallel FOMC tests across models and methods"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples per test (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing (default: 10)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel workers (default: 4)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to test (default: all 5 models)"
    )
    
    args = parser.parse_args()
    
    # Check API key
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("TOGETHERAI_API_KEY"):
        logger.error("ERROR: TOGETHERAI_API_KEY not found in environment")
        return 1
    
    # Select models
    models = args.models or TEST_MODELS
    
    # Run tests
    try:
        summary = run_parallel_tests(
            models=models,
            samples=args.samples,
            batch_size=args.batch_size,
            max_workers=args.max_workers
        )
        
        if summary["benchforge_superset"]:
            print("\n✅ SUCCESS: BenchForge is a complete superset of FLAME!")
        else:
            print("\n⚠️  WARNING: BenchForge may not be a complete superset")
        
        print(f"\nResults saved to: results/comparison/")
        return 0
        
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())