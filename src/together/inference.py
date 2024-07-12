import argparse
import re
import numpy as np
import pandas as pd
from time import time
from datetime import date
from pathlib import Path
from fpb.fpb_inference import fpb_inference
from numclaim.numclaim_inference import numclaim_inference
from fomc.fomc_inference import fomc_inference
from finbench.finbench_inference import finbench_inference
from finer.finer_inference import finer_inference
from finentity.finentity_inference import finentity_inference
from sklearn.metrics import accuracy_score

def main():
    args = parse_arguments()
    task = args.task.strip('“”"')
    
    task_inference_map = {
        "numclaim": numclaim_inference,
        "fpb": fpb_inference,
        "fomc": fomc_inference,
        "finbench": finbench_inference,
        "finer": finer_inference,
        "finentity": finentity_inference,
    }

    if task in task_inference_map:
        start_t = time()
        inference_function = task_inference_map[task]
        df = inference_function(args)
        time_taken = time() - start_t
        print(time_taken)
        results_path = (
            ROOT_DIR
            / "results"
            / task
            / f"{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)

    else:
        print(f"Task '{task}' not found in the task generation map.")


if __name__ == "__main__":
    main()
