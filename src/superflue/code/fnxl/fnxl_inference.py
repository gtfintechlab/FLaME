import time
from datetime import date
import pandas as pd
from datasets import load_dataset
from litellm import batch_completion
import json

from superflue.code.inference_prompts import fnxl_prompt
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

def fnxl_inference(args):
    today = date.today()
    logger.info(f"Starting FNXL inference (extraction+classification) on {today}")

   
    logger.info("Loading dataset...")
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

    batch_size = args.batch_size
    total_samples = len(sentences)
    total_batches = (total_samples // batch_size) + int(total_samples % batch_size > 0)

    sentence_batches = chunk_list(sentences, batch_size)
    company_batches = chunk_list(companies, batch_size)
    doc_batches = chunk_list(doc_types, batch_size)
    
    batched_llm_responses = []
    batched_complete_responses = []

    for batch_idx, (sent_batch, comp_batch, doc_batch) in enumerate(
        zip(sentence_batches, company_batches, doc_batches)
    ):
        messages_batch = []
        for snt, cpy, dtyp in zip(sent_batch, comp_batch, doc_batch):
            user_content = fnxl_prompt(snt, cpy, dtyp)
            messages_batch.append([{"role": "user", "content": user_content}])

       
        try:
            batch_responses = process_batch_with_retry(args, messages_batch, batch_idx, total_batches)
        except Exception as e:
            logger.error(f"Batch {batch_idx+1} failed: {e}")

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

    
    df = pd.DataFrame({
        "sentence": sentences,
        "company": companies,
        "docType": doc_types,
        "actual_labels": actual_labels,
        "llm_responses": batched_llm_responses,
        "complete_responses": batched_complete_responses,
    })

    results_path = RESULTS_DIR / "fnxl" / f"fnxl_extraction_{args.model}_{today.strftime('%Y%m%d')}.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Saved to {results_path}")

    return df