from superflue.together_code.fpb.fpb_evaluate import fpb_evaluate
from superflue.together_code.numclaim.numclaim_evaluate import numclaim_evaluate
from superflue.together_code.fnxl.fnxl_evaluate import fnxl_evaluate
from superflue.together_code.fomc.fomc_evaluate import fomc_evaluate
from superflue.together_code.finbench.finbench_evaluate import finbench_evaluate
from superflue.together_code.finer.finer_evaluate import finer_evaluate
from superflue.together_code.finentity.finentity_evaluate import finentity_evaluate
from superflue.together_code.headlines.headlines_evaluate import headlines_evaluate
# # from superflue.together_code.fiqa.fiqa_task1_evaluate import fiqa_evaluate
# # from superflue.together_code.fiqa.fiqa_task2_evaluate import fiqa_task2_evaluate
from superflue.together_code.edtsum.edtsum_evaluate import edtsum_evaluate
from superflue.together_code.banking77.banking77_evaluate import banking77_evaluate
from superflue.together_code.finred.finred_evaluate import finred_evaluate
from superflue.together_code.causal_classification.causal_classification_evaluate import causal_classification_evaluate
from superflue.together_code.subjectiveqa.subjectiveqa_evaluate import subjectiveqa_evaluate
from superflue.together_code.ectsum.ectsum_evaluate import ectsum_evaluate
from superflue.together_code.refind.refind_evaluate import refind_evaluate

import pandas as pd
from time import time
from datetime import date
from superflue.utils.logging_utils import setup_logger
from superflue.config import LOG_DIR, RESULTS_DIR, LOG_LEVEL, EVALUATION_DIR

logger = setup_logger(
    name="together_evaluate",
    log_file=LOG_DIR / "together_evaluate.log",
    level=LOG_LEVEL,
)


def main(args):
    task = args.dataset.strip('“”"')

    task_evaluate_map = {
        "numclaim": numclaim_evaluate,
        "fpb": fpb_evaluate,
        "fomc": fomc_evaluate,
        "finbench": finbench_evaluate,
        "finer": finer_evaluate,
        "finentity": finentity_evaluate,
        "headlines": headlines_evaluate,
        # # "fiqa_task1": fiqa_evaluate,
        # # "fiqa_task2": fiqa_task2_evaluate,
        "edtsum": edtsum_evaluate,
        "fnxl": fnxl_evaluate,
        "finred": finred_evaluate,
        "causal_classification": causal_classification_evaluate,
        "subjectiveqa": subjectiveqa_evaluate,
        "ectsum": ectsum_evaluate,
        "refind": refind_evaluate,
        "banking77": banking77_evaluate,
    }

    if task in task_evaluate_map:
        evaluate_function = task_evaluate_map[task]
        df = evaluate_function(args.file_name, args)[0]  # Pass the file_name as an additional parameter
        metrics_df = evaluate_function(args.file_name, args)[1]
        results_path = (
            EVALUATION_DIR
            / task
            / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)
        logger.info(f"Evaluation completed for {task}. Results saved to {results_path}")
        
        metrics_path = (
            EVALUATION_DIR
            / task
            / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}_metrics.csv"
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_path, index=False)
    else:
        logger.error(f"Task '{task}' not found in the task evaluation map.")
