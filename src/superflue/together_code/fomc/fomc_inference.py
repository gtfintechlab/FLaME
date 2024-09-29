import time
from datetime import date

import pandas as pd
import tqdm
from datasets import load_dataset

import together
from superflue.together_code.prompts import fomc_prompt
from superflue.together_code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="fomc_inference", log_file=LOG_DIR / "fomc_inference.log", level=LOG_LEVEL
)


def fomc_inference(args):
    today = date.today()
    logger.info(f"Starting FOMC inference on {today}")
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/fomc_communication", trust_remote_code=True)
    sentences = []
    llm_responses = []
    actual_labels = []
    complete_responses = []
    logger.info(f"Starting inference on dataset: {args.task}...")
    # start_t = time.time()

    for i in tqdm(range(len(dataset["test"])), desc="Processing sentences"):
        sentence = dataset["test"][i]["sentence"]
        actual_label = dataset["test"][i]["label"]
        sentences.append(sentence)
        actual_labels.append(actual_label)

        try:
            logger.debug(f"Processing sentence {i+1}/{len(dataset['test'])}")
            model_response = together.Complete.create(
                prompt=fomc_prompt(sentence),
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
            logger.error(f"Error processing sentence {i+1}: {e}")
            time.sleep(10.0)
            continue

    df = pd.DataFrame(
        {
            "sentences": sentences,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    results_path = (
        RESULTS_DIR
        / args.task  # (Glenn) Do we really need to use args.task if we are already running the FOMC task?
        / f"{args.task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")
    return df
