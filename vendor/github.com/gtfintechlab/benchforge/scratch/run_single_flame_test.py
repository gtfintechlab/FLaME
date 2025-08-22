#!/usr/bin/env python3
"""Run a single FLAME FOMC test for one model with 50 samples.

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
flame_root = Path(__file__).parent.parent
sys.path.insert(0, str(flame_root))

# Configure logging
def setup_logging(model_name: str):
    """Setup logging for this specific model run."""
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"flame_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - FLAME-{model_name} - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def run_flame_test(model: str, num_samples: int = 50, batch_size: int = 10):
    """Run FLAME FOMC test for a single model.
    
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
    
    logger.info(f"Starting FLAME test for {model}")
    logger.info(f"Samples: {num_samples}, Batch size: {batch_size}")
    
    try:
        # Import FLAME modules
        from flame.code.fomc.fomc_inference import fomc_inference
        from flame.code.fomc.fomc_evaluate import fomc_evaluate
        from flame.utils.dataset_utils import safe_load_dataset
        
        # Create args object
        class Args:
            pass
        
        args = Args()
        args.model = model
        args.batch_size = batch_size
        args.max_tokens = 128
        args.temperature = 0.0
        args.top_p = 0.9
        args.top_k = None
        args.repetition_penalty = 1.0
        args.prompt_format = "zero_shot"
        args.dataset = "fomc"
        args.task = "fomc"
        
        # Load dataset
        logger.info("Loading FOMC dataset...")
        dataset = safe_load_dataset("gtfintechlab/fomc_communication", trust_remote_code=True)
        test_data = dataset["test"]
        
        # Limit to num_samples
        if num_samples and num_samples < len(test_data):
            subset_data = [test_data[i] for i in range(num_samples)]
            
            # Mock the dataset loading
            import flame.code.fomc.fomc_inference as fomc_module
            original_load = fomc_module.load_fomc_dataset
            
            def mock_load():
                return subset_data
            
            fomc_module.load_fomc_dataset = mock_load
            logger.info(f"Limited dataset to {num_samples} samples")
        
        # Run inference
        logger.info("Starting inference...")
        start_time = time.time()
        
        results_df = fomc_inference(args)
        
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f}s")
        
        # Save intermediate results
        output_dir = Path("results/flame")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        temp_file = output_dir / f"{model_name}_temp.csv"
        results_df.to_csv(temp_file, index=False)
        
        # Run evaluation
        logger.info("Starting evaluation...")
        eval_results_df, metrics_df = fomc_evaluate(str(temp_file), args)
        
        # Save final results
        results_file = output_dir / f"{model_name}_50samples.csv"
        metrics_file = output_dir / f"{model_name}_metrics.csv"
        
        eval_results_df.to_csv(results_file, index=False)
        metrics_df.to_csv(metrics_file, index=False)
        
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Metrics saved to {metrics_file}")
        
        # Clean up temp file
        temp_file.unlink()
        
        # Restore original function
        if num_samples:
            fomc_module.load_fomc_dataset = original_load
        
        # Extract key metrics
        accuracy = metrics_df[metrics_df["Metric"] == "Accuracy"]["Value"].values[0]
        f1_score = metrics_df[metrics_df["Metric"] == "F1 Score"]["Value"].values[0]
        
        # Calculate extraction rate
        extracted = (eval_results_df["extracted_labels"] != -1).sum()
        extraction_rate = extracted / len(eval_results_df)
        
        logger.info(f"Test completed successfully!")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  F1 Score: {f1_score:.3f}")
        logger.info(f"  Extraction Rate: {extraction_rate:.1%}")
        
        # Save summary
        summary = {
            "model": model,
            "model_name": model_name,
            "method": "flame",
            "num_samples": num_samples,
            "accuracy": float(accuracy),
            "f1_score": float(f1_score),
            "extraction_rate": float(extraction_rate),
            "extraction_success": int(extracted),
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
            "method": "flame",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        output_dir = Path("results/flame")
        output_dir.mkdir(parents=True, exist_ok=True)
        error_file = output_dir / f"{model_name}_error.json"
        
        with open(error_file, "w") as f:
            json.dump(error_summary, f, indent=2)
        
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run FLAME FOMC test for a single model"
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
        summary = run_flame_test(args.model, args.samples, args.batch_size)
        print(f"\n✅ Test completed successfully for {summary['model_name']}")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())