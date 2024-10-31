import time
from datetime import date
import pandas as pd
from datasets import load_dataset
import together

# Custom imports for FNXL prompt and token handling
from superflue.together_code.prompts import fnxl_prompt  # Custom prompt function for FNXL
from superflue.together_code.tokens import tokens  # Custom token handling function for FNXL
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

# Setup logger for FNXL inference
logger = setup_logger(
    name="fnxl_inference",
    log_file=LOG_DIR / "fnxl_inference.log",
    level=LOG_LEVEL,
)

# Initialize the Together client
client = together.Together()

def fnxl_inference(args):
    today = date.today()
    logger.info(f"Starting FNXL inference on {today}")

    # Load the FNXL dataset (test split)
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/fnxl", trust_remote_code=True)

    results_path = (
        RESULTS_DIR
        / "fnxl"
        / f"fnxl_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize lists to store sentence information and model responses
    sentences = []
    numerals_tags = []
    companies = []
    predicted_labels = []
    actual_labels = []
    complete_responses = []

    logger.info(f"Starting inference on FNXL with model {args.model}...")

    # Iterate through the test split of the dataset
    for i in range(len(dataset["test"])):  # type: ignore
        sentence = dataset["test"][i]["sentence"]  # Extract sentence # type: ignore
        numeral_tag = dataset["test"][i]["numerals-tags"]  # Extract numeral tag # type: ignore
        company = dataset["test"][i]["company"]  # Extract company # type: ignore
        actual_label = dataset["test"][i]["ner_tags"]  # Extract actual label (NER tags) # type: ignore

        # Append to respective lists
        sentences.append(sentence)
        numerals_tags.append(numeral_tag)
        companies.append(company)
        actual_labels.append(actual_label)

        try:
            logger.info(f"Processing sentence {i+1}/{len(dataset['test'])}")  # type: ignore
            # FNXL-specific prompt to classify numerals in financial sentences
            model_response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": fnxl_prompt(sentence, numeral_tag, company)}], # type: ignore
                tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )

            # Append the model response and predicted label for the sentence
            complete_responses.append(model_response)
            predicted_label = model_response.choices[0].message.content.strip()  # type: ignore
            predicted_labels.append(predicted_label)

            logger.info(f"Model response for sentence {i+1}: {predicted_label}")

        except Exception as e:
            # Log the error and retry the same sentence after a delay
            logger.error(f"Error processing sentence {i+1}: {e}")
            time.sleep(20.0)
            complete_responses.append(None)
            predicted_labels.append(None)
            continue  # Proceed to the next sentence after sleeping

    # Create the final DataFrame after the loop
    df = pd.DataFrame(
        {
            "sentences": sentences,
            "numerals_tags": numerals_tags,
            "companies": companies,
            "predicted_labels": predicted_labels,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    # Save the results to a CSV file
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df
