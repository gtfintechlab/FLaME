import time
from datetime import date
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from litellm import completion
from superflue.code.prompts import finred_prompt

# from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.utils.path_utils import get_inference_save_path
from superflue.config import LOG_DIR, LOG_LEVEL

# Setup logger for FinRED inference
logger = setup_logger(
    name="finred_inference",
    log_file=LOG_DIR / "finred_inference.log",
    level=LOG_LEVEL,
)


def finred_inference(args):
    today = date.today()
    logger.info(f"Starting FinRED inference on {today}")

    # Load the FinRED dataset (test split)
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/FinRed", trust_remote_code=True)

    # Initialize lists to store sentences, actual labels, model responses, and complete responses
    sentences = []
    llm_responses = []
    actual_label = []
    complete_responses = []
    entities_list = []  # To store entity pairs

    logger.info(f"Starting inference on FinRED with model {args.model}...")

    # Iterate through the test split of the dataset
    for i in tqdm(range(len(dataset["test"]))):  # type: ignore
        sentence = dataset["test"][i]["sentence"]  # Extract sentence # type: ignore
        entity_pairs = dataset["test"][i][
            "entities"
        ]  # Extract entity pairs # type: ignore
        labels = dataset["test"][i][
            "relations"
        ]  # Extract the actual label (relations) # type: ignore

        # Process each entity pair in the sentence
        for entity_pair, label in zip(entity_pairs, labels):
            entity1, entity2 = entity_pair
            sentences.append(sentence)
            actual_label.append(label)
            entities_list.append((entity1, entity2))

            try:
                logger.debug(
                    f"Processing sentence {i+1}/{len(dataset['test'])}, entity pair {entity1}-{entity2}"
                )  # type: ignore

                prompt = finred_prompt(sentence, entity1, entity2)
                model_response = completion(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    # stop=tokens(args.model),
                )
                complete_responses.append(model_response)
                response_text = model_response.choices[0].message.content.strip()  # type: ignore
                llm_responses.append(response_text)

                logger.debug(
                    f"Model response for sentence {i+1}, entity pair {entity1}-{entity2}: {response_text}"
                )

            except Exception as e:
                # Log the error and retry the same sentence after a delay
                logger.error(
                    f"Error processing sentence {i+1}, entity pair {entity1}-{entity2}: {e}"
                )
                time.sleep(10.0)
                complete_responses.append(None)
                llm_responses.append(None)
                continue  # Proceed to the next sentence after sleeping

    # Create the final DataFrame after the loop
    df = pd.DataFrame(
        {
            "sentence": sentences,
            "entity_pairs": entities_list,
            "actual_label": actual_label,
            "llm_responses": llm_responses,
            "complete_responses": complete_responses,
        }
    )

    # Save results using consistent path utility
    results_path = get_inference_save_path(args.dataset, args.model)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df
