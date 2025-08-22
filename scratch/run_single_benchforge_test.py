#!/usr/bin/env python3
"""Run a single BenchForge FOMC test for one model with 50 samples.

This script is designed to be run independently for each model,
enabling parallel execution across multiple models.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup paths
benchforge_root = Path(__file__).parent
sys.path.insert(0, str(benchforge_root))

# Configure logging
def setup_logging(model_name: str):
    """Setup logging for this specific model run."""
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"benchforge_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - BF-{model_name} - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def run_benchforge_test(model: str, num_samples: int = 50, batch_size: int = 10):
    """Run BenchForge FOMC test for a single model.
    
    Args:
        model: Full model identifier (e.g., 'together_ai/meta-llama/...')
        num_samples: Number of samples to test
        batch_size: Batch size for processing
    
    Returns:
        Dictionary with test results
    """
    # Extract model name for file naming
    model_name = model.split("/")[-1].lower().replace("-instruct", "").replace("-turbo", "")
    logger = setup_logging(model_name)
    
    logger.info(f"Starting BenchForge test for {model}")
    logger.info(f"Samples: {num_samples}, Batch size: {batch_size}")
    
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
            batch_size=batch_size,
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
        
        # Set LLM client for Strategy 7
        task.set_llm_client(llm_client)
        
        # Load dataset
        logger.info("Loading FOMC dataset...")
        dataset = task.load_dataset("test")
        
        # Limit samples
        if num_samples and num_samples < len(dataset):
            samples = [dataset[i] for i in range(num_samples)]
        else:
            samples = dataset
        
        logger.info(f"Processing {len(samples)} samples")
        
        # Generate prompts
        prompts = task.process_batch(samples, config.prompt_format)
        
        # Process through LLM
        logger.info("Starting inference...")
        start_time = time.time()
        
        responses = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            batch_idx = i // batch_size + 1
            total_batches = (len(prompts) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_idx}/{total_batches}")
            batch_responses = llm_client.complete_batch(batch)
            responses.extend(batch_responses)
            
            # Small delay to avoid rate limiting
            if i + batch_size < len(prompts):
                time.sleep(0.5)
        
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f}s")
        
        # Extract labels
        logger.info("Extracting labels from responses...")
        extracted_labels = []
        for response in responses:
            if hasattr(response, 'choices'):
                text = response.choices[0].message.content if response.choices else ""
            else:
                text = str(response) if response else ""
            label = task.extract_label_from_response(text)
            extracted_labels.append(label)
        
        # Format results with evaluation
        logger.info("Formatting results and computing metrics...")
        results_dict = task.format_results_with_evaluation(
            samples, prompts, responses, extracted_labels
        )
        
        results_df = results_dict["results"]
        metrics_df = results_dict["metrics"]
        
        # Save results
        output_dir = Path("results/benchforge")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / f"{model_name}_50samples.csv"
        metrics_file = output_dir / f"{model_name}_metrics.csv"
        
        results_df.to_csv(results_file, index=False)
        metrics_df.to_csv(metrics_file, index=False)
        
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Metrics saved to {metrics_file}")
        
        # Extract key metrics
        accuracy = metrics_df[metrics_df["Metric"] == "Accuracy"]["Value"].values[0]
        f1_score = metrics_df[metrics_df["Metric"] == "F1 Score"]["Value"].values[0]
        extraction_rate = metrics_df[metrics_df["Metric"] == "Extraction Success Rate"]["Value"].values[0]
        
        logger.info(f"Test completed successfully!")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  F1 Score: {f1_score:.3f}")
        logger.info(f"  Extraction Rate: {extraction_rate:.1%}")
        
        # Save summary
        summary = {
            "model": model,
            "model_name": model_name,
            "method": "benchforge",
            "num_samples": num_samples,
            "accuracy": float(accuracy),
            "f1_score": float(f1_score),
            "extraction_rate": float(extraction_rate),
            "extraction_success": int(extraction_rate * num_samples),
            "inference_time": inference_time,
            "timestamp": datetime.now().isoformat()
        }
        
        summary_file = output_dir / f"{model_name}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        return summary
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Save error summary
        error_summary = {
            "model": model,
            "model_name": model_name,
            "method": "benchforge",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        output_dir = Path("results/benchforge")
        output_dir.mkdir(parents=True, exist_ok=True)
        error_file = output_dir / f"{model_name}_error.json"
        
        with open(error_file, "w") as f:
            json.dump(error_summary, f, indent=2)
        
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run BenchForge FOMC test for a single model"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier (e.g., 'together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo')"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to test (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Check API key
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("TOGETHERAI_API_KEY"):
        print("ERROR: TOGETHERAI_API_KEY not found in environment")
        return 1
    
    try:
        summary = run_benchforge_test(args.model, args.samples, args.batch_size)
        print(f"\n✅ Test completed successfully for {summary['model_name']}")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())