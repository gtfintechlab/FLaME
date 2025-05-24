from datetime import date
import json

import pandas as pd
from flame.utils.dataset_utils import safe_load_dataset
from tqdm import tqdm

from flame.code.prompts import get_prompt, PromptFormat
from flame.utils.logging_utils import get_component_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry

# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference", "fnxl")


def fnxl_inference(args):
    today = date.today()
    logger.info(f"Starting FNXL inference (extraction+classification) on {today}")

    logger.info("Loading dataset...")
    dataset = safe_load_dataset("gtfintechlab/fnxl", trust_remote_code=True)
    test_data = dataset["test"]  # type: ignore

    sentences = []
    companies = []
    doc_types = []
    actual_labels = []

    if args.prompt_format == "fewshot":
        fnxl_prompt = get_prompt("fnxl", PromptFormat.FEW_SHOT)
    else:
        fnxl_prompt = get_prompt("fnxl", PromptFormat.ZERO_SHOT)
    if fnxl_prompt is None:
        raise RuntimeError("FNXL prompt not found in registry")

    # 2) Iterate over the rows
    for row in test_data:
        sentence = row["sentence"]  # type: ignore
        company = row["company"]  # type: ignore
        doc_type = row["docType"]  # type: ignore

        try:
            numerals_tags_str = row["numerals-tags"]  # type: ignore
            numerals_tags_dict = json.loads(numerals_tags_str.replace("'", '"'))
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse numerals-tags JSON string: {numerals_tags_str}. Error: {e}"
            )
            numerals_tags_dict = {}

        sentences.append(sentence)
        companies.append(company)
        doc_types.append(doc_type)
        actual_labels.append(numerals_tags_dict)

    # Create batches for processing
    sentence_batches = chunk_list(sentences, args.batch_size)
    company_batches = chunk_list(companies, args.batch_size)
    doc_batches = chunk_list(doc_types, args.batch_size)
    total_batches = len(sentence_batches)

    batched_llm_responses = []
    batched_complete_responses = []

    # 4) For each batch, build a list of "messages"
    pbar = tqdm(
        enumerate(zip(sentence_batches, company_batches, doc_batches)),
        total=total_batches,
        desc="Processing FNXL entries",
    )
    for batch_idx, (sent_batch, comp_batch, doc_batch) in pbar:
        messages_batch = []
        for snt, cpy, dtyp in zip(sent_batch, comp_batch, doc_batch):
            user_content = fnxl_prompt(snt, cpy, dtyp)
            messages_batch.append([{"role": "user", "content": user_content}])

        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            for _ in messages_batch:
                batched_llm_responses.append("error")
                batched_complete_responses.append(None)
            continue

        for response in batch_responses:
            try:
                llm_text = response.choices[0].message.content.strip()  # type: ignore
                batched_llm_responses.append(llm_text)
                batched_complete_responses.append(response)
            except (KeyError, IndexError, AttributeError) as e:
                logger.error(f"Error extracting text: {e}")
                batched_llm_responses.append("error")
                batched_complete_responses.append(None)

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

    # Calculate success rate
    success_rate = (
        sum(1 for r in batched_llm_responses if r != "error")
        / len(batched_llm_responses)
    ) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
