import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
from datasets import load_dataset

import together
from src.together.prompts import finbench_prompt
from src.together.tokens import tokens

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def finbench_inference(args):
    together.api_key = args.api_key
    today = date.today()
    logger.info(f"Starting FinBench inference on {today}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/finbench", token=args.hf_token)

    # Initialize lists to store actual labels and model responses
    X_ml_data = []
    X_ml_unscale_data = []
    y_data = []
    llm_responses = []
    complete_responses = []

    logger.info("Starting inference on dataset...")
    start_t = time.time()

    # Iterating through the test split of the dataset
    for i in range(len(dataset["test"])):
        instance = dataset["test"][i]
        X_ml = instance["X_ml"]
        X_ml_unscale = instance["X_ml_unscale"]
        y = instance["y"]
        X_ml_data.append(X_ml)
        X_ml_unscale_data.append(X_ml_unscale)
        y_data.append(y)

        try:
            logger.info(f"Processing instance {i+1}/{len(dataset['test'])}")

            prompt = (
                finbench_prompt
                + f"Tabular data: {X_ml}\nProfile data: {instance['X_profile']}\nPredict the risk category:"
            )

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
            response_label = model_response["output"]["choices"][0]["text"]
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
        ROOT_DIR
        / "results"
        / args.task
        / f"{args.task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)

    logger.info(f"Inference completed. Results saved to {results_path}")
    return df
