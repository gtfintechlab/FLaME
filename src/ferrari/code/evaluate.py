from ferrari.code.fomc.fomc_evaluate import fomc_evaluate
from ferrari.code.fiqa.fiqa_task1_evaluate import fiqa_task1_evaluate
from ferrari.code.fiqa.fiqa_task2_evaluate import fiqa_task2_evaluate
from ferrari.code.mmlu.mmlu_evaluate import mmlu_evaluate
# from ferrari.code.bizbench.bizbench_evaluate import bizbench_evaluate
# from ferrari.code.econlogicqa.econlogicqa_evaluate import econlogicqa_evaluate

import pandas as pd
from time import time
from datetime import date
from ferrari.utils.logging_utils import setup_logger
from ferrari.config import LOG_DIR, RESULTS_DIR, LOG_LEVEL, EVALUATION_DIR

logger = setup_logger(
    name="together_evaluate",
    log_file=LOG_DIR / "together_evaluate.log",
    level=LOG_LEVEL,
)

def main(args):
    """Run evaluation for the specified task.
    
    Args:
        args: Command line arguments containing:
            - dataset: Name of the task/dataset
            - dataset_org: Organization holding the dataset
            - model: Model to use
            - file_name: Path to inference results
            - Other task-specific parameters
    """
    task = args.dataset.strip('"""')
    
    # Log dataset organization info
    logger.info(f"Using dataset organization: {args.dataset_org}")

    # Map of tasks to their evaluation functions
    task_evaluate_map = {
        "fomc": fomc_evaluate,
        "fiqa_task1": fiqa_task1_evaluate,
        "fiqa_task2": fiqa_task2_evaluate,
        "mmlu": mmlu_evaluate,
        # "bizbench": bizbench_evaluate,
        # "econlogicqa": econlogicqa_evaluate,
    }
    
    # Map of tasks to their dataset names
    task_dataset_map = {
        "fomc": "fomc_communication",
        "fiqa_task1": "fiqa_sentiment",
        "fiqa_task2": "fiqa_qa",
        "mmlu": "mmlu",  # MMLU uses cais/mmlu directly
        "bizbench": "bizbench",
        "econlogicqa": "econlogicqa",
    }

    if task in task_evaluate_map:
        evaluate_function = task_evaluate_map[task]
        
        # Add dataset name to args for consistent handling
        if not hasattr(args, 'dataset_name'):
            setattr(args, 'dataset_name', task_dataset_map[task])
            
        # Run evaluation
        df, metrics_df = evaluate_function(args.file_name, args)
        
        # Add dataset organization to metrics if not already present
        if "Dataset Organization" not in metrics_df["Metric"].values:
            metrics_df = pd.concat([
                metrics_df,
                pd.DataFrame({
                    "Metric": ["Dataset Organization"],
                    "Value": [args.dataset_org]
                })
            ], ignore_index=True)
        
        # Save evaluation results
        results_path = (
            EVALUATION_DIR
            / task
            / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)
        logger.info(f"Evaluation completed for {task}. Results saved to {results_path}")
        
        # Save metrics
        metrics_path = (
            EVALUATION_DIR
            / task
            / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}_metrics.csv"
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Metrics saved to {metrics_path}")
    else:
        logger.error(f"Task '{task}' not found in the task evaluation map.")