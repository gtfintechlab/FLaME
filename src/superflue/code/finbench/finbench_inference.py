import time
from datetime import date
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from superflue.code.prompts import finbench_prompt
from superflue.utils.path_utils import get_inference_path
from superflue.utils.logging_utils import get_logger
from litellm import completion

# Get logger for this module
logger = get_logger(__name__)


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

    # Iterating through the test split of the dataset
    for i in tqdm(range(len(dataset["test"])), desc="Processing sentences"):  # type: ignore
        instance = dataset["test"][i]  # type: ignore
        X_ml = instance["X_ml"]
        X_ml_unscale = instance["X_ml_unscale"]
        y = instance["y"]
        X_ml_data.append(X_ml)
        X_ml_unscale_data.append(X_ml_unscale)
        y_data.append(y)

        try:
            logger.info(f"Processing instance {i+1}/{len(dataset['test'])}")  # type: ignore

            model_response = completion(
                model=args.model,
                messages=[
                    {"role": "user", "content": finbench_prompt(instance["X_profile"])}
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                # stop=tokens(args.model)
            )
            logger.debug(f"Model response: {model_response}")
            complete_responses.append(model_response)
            response_label = model_response.choices[0].message.content  # type: ignore
            llm_responses.append(response_label)

        except Exception as e:
            logger.error(f"Error processing instance {i+1}: {e}")
            complete_responses.append(None)
            llm_responses.append(None)
            time.sleep(10.0)
            continue

    df = pd.DataFrame(
        {
            "X_ml": X_ml_data,
            "X_ml_unscale": X_ml_unscale_data,
            "actual_label": y_data,
            "llm_responses": llm_responses,
            "complete_responses": complete_responses,
        }
    )

    results_path = get_inference_path(args.dataset, args.model)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)

    return df
