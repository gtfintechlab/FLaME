from datetime import date
import pandas as pd
from datasets import load_dataset
import json
from superflue.code.inference_prompts import fnxl_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

# Setup logger for FNXL inference
logger = setup_logger(
    name="fnxl_inference",
    log_file=LOG_DIR / "fnxl_inference.log",
    level=LOG_LEVEL,
)

def fnxl_inference(args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting inference for {task} using model {args.model}.")
    dataset = load_dataset("gtfintechlab/fnxl", trust_remote_code=True)
    test_data = dataset["test"]  # type: ignore

    sentences = []
    companies = []
    doc_types = []
    actual_labels = []    
    llm_responses = []     
    complete_responses = []

    for row in test_data:
        sentence = row["sentence"] # type: ignore
        company = row["company"] # type: ignore
        doc_type = row["docType"] # type: ignore
        
        try:
            numerals_tags_str = row["numerals-tags"] # type: ignore
            numerals_tags_dict = json.loads(numerals_tags_str.replace("'", "\""))
        except:
            numerals_tags_dict = {}

        sentences.append(sentence)
        companies.append(company)
        doc_types.append(doc_type)
        actual_labels.append(numerals_tags_dict)

    sentence_batches = chunk_list(sentences, args.batch_size)
    company_batches = chunk_list(companies, args.batch_size)
    doc_batches = chunk_list(doc_types, args.batch_size)
    total_batches = len(sentence_batches)

    for batch_idx, (sent_batch, comp_batch, doc_batch) in enumerate(
        zip(sentence_batches, company_batches, doc_batches)
    ):
        messages_batch = []
        for snt, cpy, dtyp in zip(sent_batch, comp_batch, doc_batch):
            user_content = fnxl_prompt(snt, cpy, dtyp)
            messages_batch.append([{"role": "user", "content": user_content}])
       
        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx+1} failed: {e}")
            for _ in messages_batch:
                llm_responses.append("error")
                complete_responses.append(None)
            continue

        for response in batch_responses:
            try:
                llm_text = response.choices[0].message.content.strip()  # type: ignore
            except (KeyError, IndexError, AttributeError) as e:
                logger.error(f"Error extracting text: {e}")
                llm_text = "error"
            complete_responses.append(response)
            llm_responses.append(llm_text)

        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    df = pd.DataFrame({
        "sentence": sentences,
        "company": companies,
        "docType": doc_types,
        "actual_labels": actual_labels,
        "llm_responses": llm_responses,
        "complete_responses": complete_responses,
    })

    success_rate = df["llm_responses"].notnull().sum() / len(df) * 100
    logger.info(f"Success rate: {success_rate}")

    return df