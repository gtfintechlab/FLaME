from flame.code.fpb.fpb_evaluate import fpb_evaluate
from flame.code.numclaim.numclaim_evaluate import numclaim_evaluate
from flame.code.fnxl.fnxl_evaluate import fnxl_evaluate
from flame.code.fomc.fomc_evaluate import fomc_evaluate
from flame.code.finbench.finbench_evaluate import finbench_evaluate
from flame.code.finer.finer_evaluate import finer_evaluate
from flame.code.finentity.finentity_evaluate import finentity_evaluate
from flame.code.headlines.headlines_evaluate import headlines_evaluate
from flame.code.fiqa.fiqa_task1_evaluate import fiqa_task1_evaluate
from flame.code.fiqa.fiqa_task2_evaluate import fiqa_task2_evaluate
from flame.code.edtsum.edtsum_evaluate import edtsum_evaluate
from flame.code.banking77.banking77_evaluate import banking77_evaluate
from flame.code.finred.finred_evaluate import finred_evaluate
from flame.code.causal_classification.causal_classification_evaluate import (
    causal_classification_evaluate,
)
from flame.code.subjectiveqa.subjectiveqa_evaluate import subjectiveqa_evaluate
from flame.code.ectsum.ectsum_evaluate import ectsum_evaluate
from flame.code.refind.refind_evaluate import refind_evaluate
from flame.code.convfinqa.convfinqa_evaluate import convfinqa_evaluate
from flame.code.finqa.finqa_evaluate import finqa_evaluate
from flame.code.tatqa.tatqa_evaluate import tatqa_evaluate
from pathlib import Path

# from flame.code.bizbench.bizbench_evaluate import bizbench_evaluate
# from flame.code.econlogicqa.econlogicqa_evaluate import econlogicqa_evaluate
# from flame.code.causal_detection.cd_evaluate import cd_evaluate
from flame.code.causal_detection.casual_detection_evaluate import (
    causal_detection_evaluate,
)

from flame.utils.logging_utils import setup_logger
from flame.config import LOG_DIR, LOG_LEVEL

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
            - model: Model to use
            - file_name: Path to inference results
            - Other task-specific parameters
    """
    task = args.dataset.strip('"""')

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
        "tatqa": tatqa_evaluate,
        "causal_detection": causal_detection_evaluate,
    }

    if task in task_evaluate_map:
        evaluate_function = task_evaluate_map[task]

        # Run evaluation
        df, metrics_df = evaluate_function(args.file_name, args)

        # Save evaluation results
        results_path = f"evaluation_{args.file_name}"
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)
        logger.info(f"Evaluation completed for {task}. Results saved to {results_path}")

        # Save metrics
        metrics_path = Path(f"{str(results_path)[:-4]}_metrics.csv")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Metrics saved to {metrics_path}")
    else:
        logger.error(f"Task '{task}' not found in the task evaluation map.")
