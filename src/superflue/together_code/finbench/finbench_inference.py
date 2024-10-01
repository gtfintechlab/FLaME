import time
from datetime import date

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

import together
from superflue.together_code.prompts import finbench_prompt
from superflue.together_code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="finbench_inference",
    log_file=LOG_DIR / "finbench_inference.log",
    level=LOG_LEVEL,
)


def finbench_inference(args):
    today = date.today()
    logger.info(f"Starting FinBench inference on {today}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/finbench", trust_remote_code=True)

    # Initialize lists to store actual labels and model responses
    X_ml_data = []
    X_ml_unscale_data = []
    y_data = []
    llm_responses = []
    complete_responses = []

    logger.info("Starting inference on dataset...")
    # start_t = time.time()

    # Iterating through the test split of the dataset
    for i in tqdm(range(len(dataset["test"])), desc="Processing sentences"): # type: ignore
        instance = dataset["test"][i] # type: ignore
        X_ml = instance["X_ml"]
        X_ml_unscale = instance["X_ml_unscale"]
        y = instance["y"]
        X_ml_data.append(X_ml)
        X_ml_unscale_data.append(X_ml_unscale)
        y_data.append(y)

        try:
            logger.info(f"Processing instance {i+1}/{len(dataset['test'])}") # type: ignore

            prompt = finbench_prompt(instance['X_profile'])

            model_response = together.Complete.create(
                prompt=prompt,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            complete_responses.append(model_response)
            response_label = model_response["choices"][0]["text"]
            llm_responses.append(response_label)

        except Exception as e:
            logger.error(f"Error processing instance {i+1}: {e}")
            time.sleep(20.0)
            continue

    df = pd.DataFrame(
        {
            "X_ml": X_ml_data,
            "X_ml_unscale": X_ml_unscale_data,
            "y": y_data,
            "llm_responses": llm_responses,
            "complete_responses": complete_responses,
        }
    )

    results_path = (
        RESULTS_DIR
        / 'finbench/finbench_meta-llama-3.1-8b/'
        / f"{'finbench'}_{'llama-3.1-8b'}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)

    logger.info(f"Inference completed. Results saved to {results_path}")
    return df
