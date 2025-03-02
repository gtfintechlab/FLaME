from time import time
from datetime import date
# from superflue.code.fpb.fpb_inference import fpb_inference
from superflue.code.numclaim.numclaim_inference import numclaim_inference
from superflue.code.fnxl.fnxl_inference import fnxl_inference
# from superflue.code.fomc.fomc_inference import fomc_inference
# from superflue.code.finbench.finbench_inference import finbench_inference
from superflue.code.finer.finer_inference import finer_inference
from superflue.code.finentity.finentity_inference import finentity_inference
# from superflue.code.headlines.headlines_inference import headlines_inference
# from superflue.code.fiqa.fiqa_task1_inference import fiqa_task1_inference
# from superflue.code.fiqa.fiqa_task2_inference import fiqa_task2_inference
# from superflue.code.edtsum.edtsum_inference import edtsum_inference
# from superflue.code.banking77.banking77_inference import banking77_inference
# from superflue.code.finred.finred_inference import finred_inference
from superflue.code.causal_classification.causal_classification_inference import causal_classification_inference
from superflue.code.subjectiveqa.subjectiveqa_inference import subjectiveqa_inference
from superflue.code.ectsum.ectsum_inference import ectsum_inference
# from superflue.code.refind.refind_inference import refind_inference
from superflue.utils.logging_utils import setup_logger
# from superflue.code.finqa.finqa_inference import finqa_inference
# from superflue.code.tatqa.tatqa_inference import tatqa_inference
# from superflue.code.convfinqa.convfinqa_inference import convfinqa_inference
# from superflue.code.causal_detection.casual_detection_inference import casual_detection_inference
# from superflue.code.mmlu.mmlu_inference import mmlu_inference
# from superflue.code.bizbench.bizbench_inference import bizbench_inference
# from superflue.code.econlogicqa.econlogicqa_inference import econlogicqa_inference

from superflue.config import LOG_DIR, RESULTS_DIR, LOG_LEVEL

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
    
    task_inference_map = {
        "numclaim": numclaim_inference,
        # "fpb": fpb_inference,
        # "fomc": fomc_inference,
        # "finbench": finbench_inference,
        # "finqa": finqa_inference,
        "finer": finer_inference,
        # "convfinqa":convfinqa_inference,
        "finentity": finentity_inference,
        # "headlines": headlines_inference,
        # "fiqa_task1": fiqa_task1_inference, 
        # "fiqa_task2": fiqa_task2_inference,
        # "edtsum": edtsum_inference,
        # "fnxl": fnxl_inference,
        # "tatqa":tatqa_inference,
        # "causal_detection": casual_detection_inference,
        # "finred": finred_inference,
        "causal_classification": causal_classification_inference,
        "subjectiveqa": subjectiveqa_inference,
        "ectsum": ectsum_inference,
        "fnxl": fnxl_inference,
        # "refind": refind_inference,
        # "banking77": banking77_inference,
        # "mmlu": mmlu_inference,
        # "bizbench": bizbench_inference,
        # "econlogicqa": econlogicqa_inference,
    }

    if task in task_inference_map:
        start_t = time()
        inference_function = task_inference_map[task]
        df = inference_function(args)
        time_taken = time() - start_t
        logger.info(f"Time taken for inference: {time_taken}")
        results_path = (
            RESULTS_DIR
            / task
            / f"{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)
        logger.info(f"Inference completed for {task}. Results saved to {results_path}")
    else:
        logger.error(f"Task '{task}' not found in the task generation map.")
