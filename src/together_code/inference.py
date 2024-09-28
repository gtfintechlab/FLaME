import sys
from pathlib import Path
import logging

# TODO: Need to figure out how to stop this pattern I hate it
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT_DIR / 'logs'
SRC_DIRECTORY = ROOT_DIR / 'src'
DATA_DIRECTORY = ROOT_DIR / 'data'
OUTPUT_DIR = DATA_DIRECTORY / 'outputs'
RESULTS_DIR = ROOT_DIR / 'results'
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))
    
import argparse
from time import time
from datetime import date
from fpb.fpb_inference import fpb_inference
from numclaim.numclaim_inference import numclaim_inference
from fnxl.fnxl_inference import fnxl_inference
from fomc.fomc_inference import fomc_inference
from finbench.finbench_inference import finbench_inference
from finer.finer_inference import finer_inference
from finentity.finentity_inference import finentity_inference
from headlines.headlines_inference import headlines_inference
from finqa.fiqa_task1_inference import fiqa_inference
from finqa.fiqa_task2_inference import fiqa_task2_inference
from edtsum.edtsum_inference import edtsum_inference
from src.utils.logging_utils import setup_logger
from datasets import load_dataset
from src.utils.sampling_utils import sample_dataset

logger = setup_logger(name="together_inference", log_file = LOG_DIR / "together_inference.log", level=logging.DEBUG)

def main(args):
    task = args.dataset.strip('“”"')

    # # Glenn: Right now there is no need to load the dataset in the inference module because the
    # # individual inference functions below are doing the data loading
    # dataset = load_dataset(args.dataset)
    # sampled_data = sample_dataset(dataset=dataset, sample_size=args.sample_size, method=args.method, split='train')
   

    task_inference_map = {
        'numclaim': numclaim_inference,
        'fpb': fpb_inference,
        'fomc': fomc_inference,
        'finbench': finbench_inference,
        'finer': finer_inference,
        'finentity': finentity_inference,
        'headlines': headlines_inference,
        'fiqa_task1': fiqa_inference, # double check this i think it might be _task1_
        'fiqa_task2' : fiqa_task2_inference,
        'edt_sum':edtsum_inference,
        'fnxl': fnxl_inference,
    }

    if task in task_inference_map:
        start_t = time()
        inference_function = task_inference_map[task]
        df = inference_function(args)
        time_taken = time() - start_t
        print(time_taken)
        results_path = RESULTS_DIR / task / f"{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)
        logger.info(f"Inference completed for {task}. Results saved to {results_path}")
    else:
        print(f"Task '{task}' not found in the task generation map.")