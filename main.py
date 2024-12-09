import yaml
import argparse
from dotenv import load_dotenv
import os
from ferrari.code.inference import main as inference
from huggingface_hub import login
from ferrari.code.evaluate import main as evaluate
from ferrari.utils.logging_utils import setup_logger
from ferrari.config import LOG_DIR, LOG_LEVEL

# Setup logger
logger = setup_logger(
    name="main",
    log_file=LOG_DIR / "main.log",
    level=LOG_LEVEL,
)

def get_args():
    parser = argparse.ArgumentParser(description="FERRArI")
    # Core arguments
    parser.add_argument("--config", type=str, help="Path to the YAML config file.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset to use.")
    parser.add_argument("--mode", type=str, choices=["inference", "evaluate"], help="Mode to run: inference or evaluate.")
    parser.add_argument("--file_name", type=str, help="File name for evaluation (required for mode=evaluate).")
    
    # Model parameters
    parser.add_argument("--model", type=str, help="Model to use (required)")
    parser.add_argument("--max_tokens", type=int, help="Max tokens to use")
    parser.add_argument("--temperature", type=float, help="Temperature to use")
    parser.add_argument("--top_p", type=float, help="Top-p to use")
    parser.add_argument("--top_k", type=float, help="Top-k to use")
    parser.add_argument("--repetition_penalty", type=float, help="Repetition penalty to use")
    parser.add_argument("--batch_size", type=int, help="Inference batch size")
    
    # Dataset parameters
    parser.add_argument("--dataset_org", type=str, default="glennmatlin", help="Organization holding the datasets")
    parser.add_argument("--sample_size", type=int, help="Sample size for dataset")
    parser.add_argument("--method", type=str, help="Sampling method")
    parser.add_argument("--seeds", type=str, help="Random seeds as comma-separated values")
    parser.add_argument("--splits", type=str, help="Dataset splits as comma-separated values")
    
    # Other parameters
    parser.add_argument("--prompt_format", type=str, help="Version of the prompt to use")
    
    # MMLU specific arguments
    parser.add_argument(
        "--mmlu-subjects",
        nargs="+",
        help="List of MMLU subjects to evaluate (default: economics subjects)",
    )
    parser.add_argument(
        "--mmlu-split",
        choices=["dev", "validation", "test"],
        default="test",
        help="MMLU dataset split to use",
    )
    parser.add_argument(
        "--mmlu-num-few-shot",
        type=int,
        default=5,
        help="Number of few-shot examples to use for MMLU evaluation",
    )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Load environment variables first
    load_dotenv()

    # Verify HuggingFace token is set
    HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    if HUGGINGFACEHUB_API_TOKEN:
        login(token=HUGGINGFACEHUB_API_TOKEN)
        logger.info("Successfully logged in to Hugging Face Hub")
    else:
        logger.warning("Hugging Face API token not found. Please set HUGGINGFACEHUB_API_TOKEN in the environment.")

    # Parse command line arguments first
    args = get_args()
    
    # Load config file if provided
    if args.config:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
            # Convert list configs to strings for consistent handling
            if 'seeds' in config and config['seeds'] is not None:
                config['seeds'] = ','.join(map(str, config['seeds']))
            if 'splits' in config and config['splits'] is not None:
                config['splits'] = ','.join(map(str, config['splits']))
            
            # Set config values as defaults
            for key, value in config.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)

    # Convert string lists back to actual lists
    if hasattr(args, 'seeds') and args.seeds:
        args.seeds = [int(x.strip()) for x in args.seeds.split(',')]
    if hasattr(args, 'splits') and args.splits:
        args.splits = [int(x.strip()) for x in args.splits.split(',')]

    # Validate required arguments
    if not args.mode or args.mode not in ['inference', 'evaluate']:
        raise ValueError("Mode is required and must be either 'inference' or 'evaluate'.")
    if args.mode == "evaluate" and not args.file_name:
        raise ValueError("File name is required for evaluation mode.")
    if not args.dataset:
        raise ValueError("Dataset is required.")
    if not args.model:
        raise ValueError("Model is required.")
    if not args.dataset_org:
        raise ValueError("Dataset organization is required.")

    # Execute the appropriate mode
    if args.mode == "inference":
        inference(args)
    elif args.mode == "evaluate":
        evaluate(args)