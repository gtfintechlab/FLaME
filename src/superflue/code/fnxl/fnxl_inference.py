import time
from datetime import date
import pandas as pd
from datasets import load_dataset
from litellm import completion 
import json

# Custom imports for FNXL prompt and token handling
from superflue.code.prompts import fnxl_prompt  # Custom prompt function for FNXL
from superflue.code.tokens import tokens  # Custom token handling function for FNXL
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

# Setup logger for FNXL inference
logger = setup_logger(
    name="fnxl_inference",
    log_file=LOG_DIR / "fnxl_inference.log",
    level=LOG_LEVEL,
)

def fnxl_inference(args):
    def flatten_nested_list(nested_list):
        """Flatten a deeply nested list into a single list."""
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(flatten_nested_list(item))  # Recursively flatten
            else:
                try:
                    flat_list.append(float(item))  # Convert to float
                except ValueError:
                    continue  # Skip non-numeric items
        return flat_list
    
    def save_progress(df, path):
        """Save the current progress to a CSV file."""
        df.to_csv(path, index=False)
        logger.info(f"Progress saved to {path}")
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
    actual_labels = []
    doc_types = []
    llm_responses = []
    complete_responses = []

    logger.info(f"Starting inference on FNXL with model {args.model}...")

    # Iterate through the test split of the dataset
    for i in range(len(dataset["test"])):  # type: ignore
        try:
            sentence = dataset["test"][i]["sentence"]  # Extract sentence # type: ignore
            # numeral_tag = json.loads(dataset["test"][i]["numerals-tags"]).values().tolist()  # Extract numeral tag # type: ignore
            try:
                numerals_tags_str = dataset["test"][i]["numerals-tags"]  # type: ignore
                numerals_tags_dict = json.loads(numerals_tags_str.replace("'", '"'))  # Replace single quotes for valid JSON
                numerals_tag = list(numerals_tags_dict.values())
                numerals_tag = flatten_nested_list(numerals_tag)
            except json.decoder.JSONDecodeError:
                numerals_tag = [] # type: ignore
            
            company = dataset["test"][i]["company"]  # Extract company # type: ignore
            # actual_label = dataset["test"][i]["ner_tags"]  # Extract actual label (NER tags) # type: ignore
            doc_type = dataset["test"][i]["docType"]  # Extract document type # type: ignore
            actual_label = numerals_tags_dict if numerals_tags_dict else {}

            # Append to respective lists
            sentences.append(sentence)
            numerals_tags.append(numerals_tag)
            companies.append(company)
            doc_types.append(doc_type)
            actual_labels.append(actual_label)

 
            logger.info(
                        f"Iteration {i+1}: Lengths - sentences: {(sentences)}, numerals_tags: {(numerals_tags)}, companies: {len(companies)}, doc_types: {len(doc_types)}, llm_responses: {len(llm_responses)}, actual_labels: {(actual_labels)}, complete_responses: {len(complete_responses)}"
                    )

            # FNXL-specific prompt to classify numerals in financial sentences
            model_response = completion(
                model=args.model,
                messages=[{"role": "user", "content": fnxl_prompt(sentence, numerals_tag, company, doc_type)}], # type: ignore
                tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )

            # Append the model response and predicted label for the sentence
            complete_responses.append(model_response)
            llm_response = model_response.choices[0].message.content.strip()  # type: ignore
            llm_responses.append(llm_response)

            logger.info(f"Model response for sentence {i+1}: {llm_response}")

        except Exception as e:
            # Log the error and retry the same sentence after a delay
            logger.error(f"Error processing sentence {i+1}: {e}")
            time.sleep(10.0)
            sentences.append(sentence if 'sentence' in locals() else None)
            numerals_tags.append([])  # Default to empty list
            companies.append(company if 'company' in locals() else None)
            doc_types.append(doc_type if 'doc_type' in locals() else None)
            actual_labels.append({})
            complete_responses.append(None)
            llm_responses.append(None)
            llm_responses.append(None)
            continue  # Proceed to the next sentence after sleeping
        if i % 10 == 0:
                assert len(sentences) == len(numerals_tags) == len(companies) == len(doc_types), "List lengths are mismatched!"

                df_progress = pd.DataFrame(
                    {
                        "sentences": sentences,
                        "numerals_tags": numerals_tags,
                        "companies": companies,
                        "doc_types": doc_types,
                        "llm_responses": llm_responses,
                        "actual_labels": actual_labels,
                        "complete_responses": complete_responses,
                    }
                )
                save_progress(df_progress, results_path)

    # Create the final DataFrame after the loop
    df = pd.DataFrame(
        {
            "sentences": sentences,
            "numerals_tags": numerals_tags,
            "companies": companies,
            "doc_types": doc_types,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    # Save the results to a CSV file
    save_progress(df, results_path)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df
