"""Inference module for all tasks."""
from ferrari.code.fomc.fomc_inference import fomc_inference
from ferrari.code.fiqa.fiqa_task1_inference import fiqa_task1_inference
from ferrari.code.fiqa.fiqa_task2_inference import fiqa_task2_inference
from ferrari.code.mmlu.mmlu_inference import mmlu_inference
from ferrari.code.bizbench.bizbench_inference import bizbench_inference
from ferrari.code.econlogicqa.econlogicqa_inference import econlogicqa_inference

from ferrari.utils.logging_utils import setup_logger
from ferrari.config import LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="together_inference",
    log_file=LOG_DIR / "together_inference.log",
    level=LOG_LEVEL,
)

def main(args):
    """Run inference for the specified task.
    
    Args:
        args: Command line arguments containing:
            - dataset: Name of the task/dataset
            - dataset_org: Organization holding the dataset
            - model: Model to use
            - Other task-specific parameters
    """
    task = args.dataset.strip('"""')
    
    # Log dataset organization info
    logger.info(f"Using dataset organization: {args.dataset_org}")
    
    # Map of tasks to their inference functions
    task_inference_map = {
        "fomc": fomc_inference,
        "fiqa_task1": fiqa_task1_inference,
        "fiqa_task2": fiqa_task2_inference,
        "mmlu": mmlu_inference,
        "bizbench": bizbench_inference,
        "econlogicqa": econlogicqa_inference,
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
    
    if task in task_inference_map:
        inference_function = task_inference_map[task]
        
        # Add dataset name to args for consistent handling
        if not hasattr(args, 'dataset_name'):
            setattr(args, 'dataset_name', task_dataset_map[task])
            
        # Run inference
        df = inference_function(args)
        logger.info(f"Inference completed for {task}.")
    else:
        logger.error(f"Task '{task}' not found in the task inference map.")