from datetime import date
import pandas as pd
from datasets import load_dataset
import json

from flame.code.prompts_zeroshot import fnxl_zeroshot_prompt
from flame.code.prompts_fewshot import fnxl_fewshot_prompt

# from flame.code.tokens import tokens
from flame.utils.logging_utils import setup_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

# Setup logger for FNXL inference
logger = setup_logger(
    name="fnxl_inference",
    log_file=LOG_DIR / "fnxl_inference.log",
    level=LOG_LEVEL,
)


def fnxl_inference(args):
    today = date.today()
    logger.info(f"Starting FNXL inference (extraction+classification) on {today}")

    # 1) Load the FNXL dataset (test split)
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/fnxl", trust_remote_code=True)
    test_data = dataset["test"]  # type: ignore

    sentences = []
    companies = []
    doc_types = []
    actual_labels = []

    if args.prompt_format == "fewshot":
        fnxl_prompt = fnxl_fewshot_prompt
    elif args.prompt_format == "zeroshot":
        fnxl_prompt = fnxl_zeroshot_prompt

    # 2) Iterate over the rows
    for row in test_data:
        sentence = row["sentence"]  # type: ignore
        company = row["company"]  # type: ignore
        doc_type = row["docType"]  # type: ignore

        # Ground truth: parse "numerals-tags" into a dict, e.g.:
        #   "{'us-gaap:SomeTag': ['7.2', '9.0']}" => {"us-gaap:SomeTag": ["7.2", "9.0"]}
        # We'll store the string version in the DF for now, or parse fully for later.
        try:
            numerals_tags_str = row["numerals-tags"]  # type: ignore
            numerals_tags_dict = json.loads(numerals_tags_str.replace("'", '"'))
        except Exception:
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
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
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
    df = pd.DataFrame(
        {
            "sentence": sentences,
            "company": companies,
            "docType": doc_types,
            "actual_labels": actual_labels,
            "llm_responses": batched_llm_responses,
            "complete_responses": batched_complete_responses,
        }
    )

    # 8) Save
    results_path = (
        RESULTS_DIR
        / "fnxl"
        / f"fnxl_extraction_{args.model}_{today.strftime('%Y%m%d')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Saved to {results_path}")

    return df
