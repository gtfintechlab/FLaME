from superflue.code.fpb.fpb_evaluate import fpb_evaluate
from superflue.code.numclaim.numclaim_evaluate import numclaim_evaluate
from superflue.code.fnxl.fnxl_evaluate import fnxl_evaluate
from superflue.code.fomc.fomc_evaluate import fomc_evaluate
from superflue.code.finbench.finbench_evaluate import finbench_evaluate
from superflue.code.finer.finer_evaluate import finer_evaluate
from superflue.code.finentity.finentity_evaluate import finentity_evaluate
from superflue.code.headlines.headlines_evaluate import headlines_evaluate
from superflue.code.fiqa.fiqa_task1_evaluate import fiqa_task1_evaluate
from superflue.code.fiqa.fiqa_task2_evaluate import fiqa_task2_evaluate
from superflue.code.edtsum.edtsum_evaluate import edtsum_evaluate
from superflue.code.banking77.banking77_evaluate import banking77_evaluate
from superflue.code.finred.finred_evaluate import finred_evaluate
from superflue.code.causal_classification.causal_classification_evaluate import causal_classification_evaluate
from superflue.code.subjectiveqa.subjectiveqa_evaluate import subjectiveqa_evaluate
from superflue.code.ectsum.ectsum_evaluate import ectsum_evaluate
from superflue.code.refind.refind_evaluate import refind_evaluate
from superflue.code.convfinqa.convfinqa_evaluate import convfinqa_evaluate
from superflue.code.finqa.finqa_evaluate import finqa_evaluate
from superflue.code.tatqa.tatqa_evaluate import tatqa_evaluate
# from superflue.code.mmlu.mmlu_evaluate import mmlu_evaluate
# from superflue.code.bizbench.bizbench_evaluate import bizbench_evaluate
# from superflue.code.econlogicqa.econlogicqa_evaluate import econlogicqa_evaluate
# from superflue.code.causal_detection.cd_evaluate import cd_evaluate

import pandas as pd
from time import time
from datetime import date
from superflue.utils.logging_utils import setup_logger
from superflue.config import LOG_DIR, RESULTS_DIR, LOG_LEVEL, EVALUATION_DIR
from pathlib import Path

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
        "numclaim": numclaim_evaluate,
        "fpb": fpb_evaluate,
        "fomc": fomc_evaluate,
        "finbench": finbench_evaluate,
        "finer": finer_evaluate,
        "finentity": finentity_evaluate,
        "headlines": headlines_evaluate,
        "fiqa_task1": fiqa_task1_evaluate,
        "fiqa_task2": fiqa_task2_evaluate,
        "edtsum": edtsum_evaluate,
        "fnxl": fnxl_evaluate,
        "finred": finred_evaluate,
        "causal_classification": causal_classification_evaluate,
        "subjectiveqa": subjectiveqa_evaluate,
        "ectsum": ectsum_evaluate,
        "refind": refind_evaluate,
        "banking77": banking77_evaluate,
        "convfinqa": convfinqa_evaluate,
        "finqa": finqa_evaluate,
        "tatqa": tatqa_evaluate
        # cd evaluate here
    }

    if task in task_evaluate_map:
        evaluate_function = task_evaluate_map[task]
        
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
        results_path = f"evaluation_{args.file_name}"
        results_path = Path(results_path)
        # results_path = (
        #     EVALUATION_DIR
        #     / task
        #     / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
        # )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)
        logger.info(f"Evaluation completed for {task}. Results saved to {results_path}")
        
        # Save metrics
        metrics_path = Path(f"{str(results_path)[:-4]}_metrics.csv")
        # metrics_path = (
        #     EVALUATION_DIR
        #     / task
        #     / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}_metrics.csv"
        # )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Metrics saved to {metrics_path}")
    else:
        logger.error(f"Task '{task}' not found in the task evaluation map.")