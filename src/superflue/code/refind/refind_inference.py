import time
from datetime import date
import pandas as pd
from datasets import load_dataset

from litellm import completion 
from superflue.code.prompts import refind_prompt
from superflue.code.tokens import tokens

from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL
from tqdm import tqdm

# Setup logger for ReFinD inference
logger = setup_logger(
    name="refind_inference",
    log_file=LOG_DIR / "refind_inference.log",
    level=LOG_LEVEL,
)

def refind_inference(args):
    
    today = date.today()
    logger.info(f"Starting ReFinD inference on {today}")

    # Load the ReFinD dataset (test split)
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/ReFinD", trust_remote_code=True)

    results_path = (
        RESULTS_DIR
        / "refind"
        / f"refind_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize lists to store entities, actual labels, model responses, and complete responses
    sentences = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    logger.info(f"Starting inference on ReFinD with model {args.model}...")

    # Iterate through the test split of the dataset
    for i in tqdm(range(len(dataset["test"]))):  # type: ignore
        sample = dataset["test"][i]  # type: ignore
        sentence = ' '.join(['[ENT1]'] + sample['token'][sample['e1_start']:sample['e1_end']] + ['[/ENT1]'] + sample['token'][sample['e1_end']+1:sample['e2_start']] + ['[ENT2]'] + sample['token'][sample['e2_start']:sample['e2_end']] + ['[/ENT2]'])
        actual_label = dataset["test"][i]["rel_group"]  # Extract the actual label (response) # type: ignore
        sentences.append(sentence)
        actual_labels.append(actual_label)

        try:
            logger.debug(f"Processing sentence {i+1}/{len(dataset['test'])}")  # type: ignore
            model_response = completion(
                model=args.model,
                messages=[{"role": "user", "content": refind_prompt(sentence)}],
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
            # print(f"Model response for sentence {i+1}: {response_text}")
            # print(f"Actual label for sentence {i+1}: {actual_label}")

            logger.debug(f"Model response for sentence {i+1}: {response_text}")

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