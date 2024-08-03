import together
import pandas as pd
import time
from datasets import load_dataset
from datetime import date
from src.together.prompts import causal_classification_prompt
from pathlib import Path
import logging
from src.together.models import get_model_name
from src.together.tokens import tokens

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def causal_classification_inference(args):
    together.api_key = args.api_key
    today = date.today()
    logger.info(f"Starting Causal Classification inference on {today}")

    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/CausalClassification", token=args.hf_token)

    # Initialize lists to store actual labels and model responses
    texts = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    logger.info(f"Starting inference on {args.task}...")
    start_t = time.time()
    for i in range(len(dataset["test"])): # type: ignore
        text = dataset["test"][i]["text"] # type: ignore
        actual_label = dataset["test"][i]["label"] # type: ignore
        texts.append(text)
        actual_labels.append(actual_label)
        try:
            logger.info(f"Processing text {i+1}/{len(dataset['test'])}") # type: ignore
            model_response = together.Complete.create(
                prompt=causal_classification_prompt(text),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            complete_responses.append(model_response)
            response_label = model_response["output"]["choices"][0]["text"] # type: ignore
            llm_responses.append(response_label)

            df = pd.DataFrame(
                {
                    "texts": texts,
                    "llm_responses": llm_responses,
                    "actual_labels": actual_labels,
                    "complete_responses": complete_responses,
                }
            )

        except Exception as e:
            logger.error(f"Error processing text {i+1}: {e}")
            time.sleep(20.0)
            continue

    results_path = (
        ROOT_DIR
        / "results"
        / args.task
        / f"{args.task}_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df
