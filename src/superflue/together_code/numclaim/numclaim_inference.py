import time
from datetime import date
import pandas as pd
from datasets import load_dataset

import together
from superflue.together_code.prompts import numclaim_prompt
from superflue.together_code.tokens import tokens

from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

# Setup logger for Numclaim inference
logger = setup_logger(
    name="numclaim_inference",
    log_file=LOG_DIR / "numclaim_inference.log",
    level=LOG_LEVEL,
)

# Initialize the Together client
client = together.Together()

def numclaim_inference(args):
    
    today = date.today()
    logger.info(f"Starting Numclaim inference on {today}")

    # Load the Numclaim dataset (test split)
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/Numclaim", trust_remote_code=True)

    results_path = (
        RESULTS_DIR
        / "numclaim"
        / f"numclaim_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize lists to store sentences, actual labels, model responses, and complete responses
    sentences = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    logger.info(f"Starting inference on Numclaim with model {args.model}...")

    # Iterate through the test split of the dataset
    for i in range(len(dataset["test"])):  # type: ignore
        sentence = dataset["test"][i]["context"]  # Extract context (sentence) # type: ignore
        actual_label = dataset["test"][i]["response"]  # Extract the actual label (response) # type: ignore
        sentences.append(sentence)
        actual_labels.append(actual_label)

        try:
            logger.info(f"Processing sentence {i+1}/{len(dataset['test'])}")  # type: ignore
            # Generate the model's response using Together API
            model_response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": numclaim_prompt(sentence)}],
                tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )

            # Append the model response and complete response for the sentence
            complete_responses.append(model_response)
            response_text = model_response.choices[0].message.content.strip()  # type: ignore
            llm_responses.append(response_text)

            logger.info(f"Model response for sentence {i+1}: {response_text}")

        except Exception as e:
            # Log the error and retry the same sentence after a delay
            logger.error(f"Error processing sentence {i+1}: {e}")
            time.sleep(10.0)
            complete_responses.append(None)
            llm_responses.append(None)
            continue  # Proceed to the next sentence after sleeping

    # Create the final DataFrame after the loop
    df = pd.DataFrame(
        {
            "sentences": sentences,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    # Save the results to a CSV file
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df
