from time import time
from datetime import date
from superflue.together_code.fpb.fpb_inference import fpb_inference
from superflue.together_code.numclaim.numclaim_inference import numclaim_inference
from superflue.together_code.fnxl.fnxl_inference import fnxl_inference
from superflue.together_code.fomc.fomc_inference import fomc_inference
from superflue.together_code.finbench.finbench_inference import finbench_inference
from superflue.together_code.finer.finer_inference import finer_inference
from superflue.together_code.finentity.finentity_inference import finentity_inference
from superflue.together_code.headlines.headlines_inference import headlines_inference
from superflue.together_code.fiqa.fiqa_task1_inference import fiqa_inference
from superflue.together_code.fiqa.fiqa_task2_inference import fiqa_task2_inference
from superflue.together_code.edtsum.edtsum_inference import edtsum_inference
from superflue.together_code.causal_classification.causal_classification_inference import causal_classification_inference
from superflue.together_code.subjectiveqa.subjectiveqa_inference import subjectiveqa_inference
from superflue.together_code.ectsum.ectsum_inference import ectsum_inference

from superflue.utils.logging_utils import setup_logger

from superflue.config import LOG_DIR, RESULTS_DIR, LOG_LEVEL

logger = setup_logger(
    name="together_inference",
    log_file=LOG_DIR / "together_inference.log",
    level=LOG_LEVEL,
)


def main(args):
    task = args.dataset.strip('“”"')

    # # Glenn: Right now there is no need to load the dataset in the inference module because the
    # # individual inference functions below are doing the data loading
    # dataset = load_dataset(args.dataset, trust_remote_code=True)
    # sampled_data = sample_dataset(dataset=dataset, sample_size=args.sample_size, method=args.method, split='train')

    task_inference_map = {
        "numclaim": numclaim_inference,
        "fpb": fpb_inference,
        "fomc": fomc_inference,
        "finbench": finbench_inference,
        "finer": finer_inference,
        "finentity": finentity_inference,
        "headlines": headlines_inference,
        "fiqa_task1": fiqa_inference,  # double check this i think it might be _task1_
        "fiqa_task2": fiqa_task2_inference,
        "edt_sum": edtsum_inference,
        "fnxl": fnxl_inference,
        "causal_classification": causal_classification_inference,
        "subjectiveqa": subjectiveqa_inference,
        "ectsum": ectsum_inference,
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
