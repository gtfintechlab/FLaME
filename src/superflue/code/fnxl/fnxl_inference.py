import time
from datetime import date
import pandas as pd
from datasets import load_dataset
from litellm import batch_completion
import json

from superflue.code.prompts_zeroshot import fnxl_zeroshot_prompt
from superflue.code.prompts_fewshot import fnxl_fewshot_prompt
# from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

# Setup logger for FNXL inference
logger = setup_logger(
    name="fnxl_inference",
    log_file=LOG_DIR / "fnxl_inference.log",
    level=LOG_LEVEL,
)

# def fnxl_inference(args):
#     def flatten_nested_list(nested_list):
#         """Flatten a deeply nested list into a single list."""
#         flat_list = []
#         for item in nested_list:
#             if isinstance(item, list):
#                 flat_list.extend(flatten_nested_list(item))  # Recursively flatten
#             else:
#                 try:
#                     flat_list.append(float(item))  # Convert to float
#                 except ValueError:
#                     continue  # Skip non-numeric items
#         return flat_list

#     today = date.today()
#     logger.info(f"Starting FNXL inference on {today}")

#     # Load the FNXL dataset (test split)
#     logger.info("Loading dataset...")
#     dataset = load_dataset("gtfintechlab/fnxl", trust_remote_code=True)

#     # Extract relevant fields from the dataset
#     sentences = []
#     companies = []
#     doc_types = []
#     actual_labels = []   # Will store dictionary ground-truth

#     for row in dataset["test"]: # type: ignore
#         # Basic fields
#         sentence = row["sentence"] # type: ignore
#         company = row["company"] # type: ignore
#         doc_type = row["docType"] # type: ignore
#         # Possibly more as needed
#         sentences.append(sentence)
#         companies.append(company)
#         doc_types.append(doc_type)

#         # Instead of flattening, keep as dictionary
#         try:
#             numerals_tags_str = row["numerals-tags"]  # type: ignore # e.g. "{'us-gaap:SomeTag': ['13']}"
#             # Convert single quotes to double quotes for valid JSON
#             numerals_tags_dict = json.loads(numerals_tags_str.replace("'", "\""))
#             # Optionally convert string numerals to floats here:
#             for k, v in numerals_tags_dict.items():
#                 # v might be a list of strings like ["13", "25"] -> convert each
#                 numerals_tags_dict[k] = [float(x) for x in v if x.replace('.', '').isdigit()]

#             actual_labels.append(numerals_tags_dict)

#         except json.decoder.JSONDecodeError:
#             # if something fails, store empty dictionary
#             actual_labels.append({})
#     llm_responses = []
#     complete_responses = []

#     batch_size = 10
#     total_batches = len(sentences) // batch_size + int(len(sentences) % batch_size > 0)
#     logger.info(f"Processing {len(sentences)} sentences in {total_batches} batches.")

#     # Create batches
#     sentence_batches = chunk_list(sentences, batch_size)
#     numeral_tag_batches = chunk_list(actual_labels, batch_size)
#     company_batches = chunk_list(companies, batch_size)
#     doc_type_batches = chunk_list(doc_types, batch_size)

#     for batch_idx, (sentence_batch, numeral_tag_batch, company_batch, doc_type_batch) in enumerate(
#             zip(sentence_batches, numeral_tag_batches, company_batches, doc_type_batches)
#         ):
#         # Create prompt messages for the batch
#         messages_batch = [
#             [{"role": "user", "content": fnxl_prompt(sentence, numeral_tag, company, doc_type)}] # type: ignore
#             for sentence, numeral_tag, company, doc_type in zip(
#                 sentence_batch, numeral_tag_batch, company_batch, doc_type_batch
#             )
#         ]

#         try:
#             # Process the batch
#             batch_responses = process_batch_with_retry(args, messages_batch, batch_idx, total_batches)

#             for response in batch_responses:
#                 try:
#                     llm_response = response.choices[0].message.content.strip()  # type: ignore
#                     llm_responses.append(llm_response)
#                     complete_responses.append(response)
#                 except (KeyError, IndexError, AttributeError) as e:
#                     logger.error(f"Error extracting response: {e}")
#                     llm_responses.append("error")
#                     complete_responses.append(None)

#         except Exception as e:
#             logger.error(f"Batch {batch_idx + 1} failed: {e}")
#             llm_responses.extend(["error"] * len(sentence_batch))
#             complete_responses.extend([None] * len(sentence_batch))
#             continue

#     # Create the final DataFrame
#     df = pd.DataFrame(
#         {
#             "sentences": sentences,
#             # "numerals_tags": numerals_tags,
#             "companies": companies,
#             "doc_types": doc_types,
#             "llm_responses": llm_responses,
#             "actual_labels": actual_labels,
#             "complete_responses": complete_responses,
#         }
#     )

#     # Save the results to a CSV file
#     results_path = (
#         RESULTS_DIR
#         / "fnxl"
#         / f"fnxl_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
#     )
#     results_path.parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(results_path, index=False)
#     logger.info(f"Inference completed. Results saved to {results_path}")

#     return df
def fnxl_inference(args):
    today = date.today()
    logger.info(f"Starting FNXL inference (extraction+classification) on {today}")

    # 1) Load the FNXL dataset (test split)
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/fnxl", trust_remote_code=True)
    test_data = dataset["test"]  # type: ignore

    # We will store lists to construct a DataFrame
    sentences = []
    companies = []
    doc_types = []
    actual_labels = []      # The ground truth dictionary from 'numerals-tags'
    llm_responses = []      # Raw JSON strings from the LLM
    complete_responses = [] # Full response objects (for debugging, optional)

    if args.prompt_format == "fewshot":
        fnxl_prompt = fnxl_fewshot_prompt
    elif args.prompt_format == "zeroshot":
        fnxl_prompt = fnxl_zeroshot_prompt

    # 2) Iterate over the rows
    for row in test_data:
        sentence = row["sentence"] # type: ignore
        company = row["company"] # type: ignore
        doc_type = row["docType"] # type: ignore

        # Ground truth: parse "numerals-tags" into a dict, e.g.:
        #   "{'us-gaap:SomeTag': ['7.2', '9.0']}" => {"us-gaap:SomeTag": ["7.2", "9.0"]}
        # We'll store the string version in the DF for now, or parse fully for later.
        try:
            numerals_tags_str = row["numerals-tags"] # type: ignore
            numerals_tags_dict = json.loads(numerals_tags_str.replace("'", "\""))
        except:
            numerals_tags_dict = {}

        sentences.append(sentence)
        companies.append(company)
        doc_types.append(doc_type)
        actual_labels.append(numerals_tags_dict)

    # 3) Prepare for LLM calls in batches
    batch_size = args.batch_size  # or 10, or whatever you like
    total_samples = len(sentences)
    total_batches = (total_samples // batch_size) + int(total_samples % batch_size > 0)

    sentence_batches = chunk_list(sentences, batch_size)
    company_batches = chunk_list(companies, batch_size)
    doc_batches = chunk_list(doc_types, batch_size)

    # We'll store results in parallel lists
    batched_llm_responses = []
    batched_complete_responses = []

    # 4) For each batch, build a list of "messages"
    for batch_idx, (sent_batch, comp_batch, doc_batch) in enumerate(
        zip(sentence_batches, company_batches, doc_batches)
    ):
        messages_batch = []
        for snt, cpy, dtyp in zip(sent_batch, comp_batch, doc_batch):
            user_content = fnxl_prompt(snt, cpy, dtyp)
            # Each request to the LLM is a list of role-content dicts
            messages_batch.append([{"role": "user", "content": user_content}])

        # 5) Call the LLM
        try:
            batch_responses = process_batch_with_retry(args, messages_batch, batch_idx, total_batches)
        except Exception as e:
            logger.error(f"Batch {batch_idx+1} failed: {e}")
            # Put placeholders
            for _ in messages_batch:
                batched_llm_responses.append("error")
                batched_complete_responses.append(None)
            continue

        # 6) Parse out the text
        for response in batch_responses:
            try:
                llm_text = response.choices[0].message.content.strip()  # type: ignore
                batched_llm_responses.append(llm_text)
                batched_complete_responses.append(response)
            except (KeyError, IndexError, AttributeError) as e:
                logger.error(f"Error extracting text: {e}")
                batched_llm_responses.append("error")
                batched_complete_responses.append(None)

    # 7) Combine all results into a DataFrame
    df = pd.DataFrame({
        "sentence": sentences,
        "company": companies,
        "docType": doc_types,
        "actual_labels": actual_labels,
        "llm_responses": batched_llm_responses,
        "complete_responses": batched_complete_responses,
    })

    # 8) Save
    results_path = RESULTS_DIR / "fnxl" / f"fnxl_extraction_{args.model}_{today.strftime('%Y%m%d')}.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Saved to {results_path}")

    return df