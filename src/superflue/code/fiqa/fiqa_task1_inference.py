import pandas as pd
from datasets import load_dataset
from superflue.utils.batch_utils import process_batch_with_retry, chunk_list
from superflue.code.inference_prompts import fiqa_task1_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL
from tqdm import tqdm

# Set up logger
logger = setup_logger(
    name="fiqa_task1_inference",
    log_file=LOG_DIR / "fiqa_task1_inference.log",
    level=LOG_LEVEL,
)

def fiqa_task1_inference(args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting inference for {task} using model {args.model}.")
    dataset = load_dataset("gtfintechlab/FiQA_Task1", trust_remote_code=True)

    test_data = dataset["test"] # type: ignore
    all_texts = [f"Sentence: {data['sentence']}. Snippets: {data['snippets']}. Target aspect: {data['target']}" for data in test_data] # type: ignore
    all_targets = [data["target"] for data in test_data] # type: ignore
    all_sentiments = [data["sentiment_score"] for data in test_data] # type: ignore

    sentence_batches = chunk_list(all_texts, args.batch_size)
    total_batches = len(sentence_batches)
    llm_responses = []
    complete_responses = []

    pbar = tqdm(sentence_batches, desc="Processing batches")
    for batch_idx, sentence_batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": fiqa_task1_prompt(sentence)}]
            for sentence in sentence_batch
        ]
        try:
            # Process batch with retry mechanism
            batch_responses = process_batch_with_retry(args, messages_batch, batch_idx, total_batches)

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in range(len(sentence_batch)):
                llm_responses.append(None)
                complete_responses.append(None)
            continue
        
        for response in batch_responses:
            try:
                response_label = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            llm_responses.append(response_label)
            complete_responses.append(response)
        
        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")
    
    # Create DataFrame with results
    df = pd.DataFrame(
        {
            "context": all_texts,
            "llm_responses": llm_responses,
            "actual_target": all_targets,
            "actual_sentiment": all_sentiments,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df['llm_responses'].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
