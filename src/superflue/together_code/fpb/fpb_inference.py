import time
from datetime import date
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from superflue.together_code.prompts import fpb_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.config import LOG_DIR
logger = setup_logger("fpb_inference", LOG_DIR / "fpb_inference.log")
import yaml

def prepare_batch(batch: Dict[str, List], args) -> List[str]:
    prompts = []
    for sentence in batch['sentence']:
        try:
            prompt = fpb_prompt(
                sentence=sentence, prompt_format=args.prompt_format
            )
            prompts.append(prompt)
        except Exception as e:
            logger.error(
                f"Error preparing prompt for sentence: {sentence}. Error: {str(e)}"
            )
            prompts.append(None)
    return prompts


def process_batch_response(
    batch_response: Dict[str, Any],
    batch: Dict[str, List],
    task: str,
    model: str,
) -> List[Dict[str, Any]]:
    results = []
    for i, choice in enumerate(batch_response["output"]["choices"]):
        try:
            result = {
                "sentence": batch["sentence"][i],
                "actual_label": batch["label"][i],
                "llm_response": choice["text"],
                "complete_response": {
                    "task": task,
                    "model": model,
                    "response": choice,
                    "metadata": {"timestamp": batch_response["output"]["created"]},
                },
            }
            results.append(result)
        except Exception as e:
            logger.error(
                f"Error processing response for sentence: {batch['sentence'][i]}. Error: {str(e)}"
            )
            results.append(
                {
                    "sentence": batch["sentence"][i],
                    "actual_label": batch["label"][i],
                    "llm_response": "error",
                    "complete_response": str(e),
                }
            )
    return results


def fpb_inference(args, process_api_call, process_api_response):
    with open(args.config, "r") as file:
        logger.debug(file)
        config = yaml.safe_load(file)
    total_time = 0
    total_batches = 0
    logger.info(f"Starting FPB inference on {date.today()}")
    # TODO: data_splits should be defined in the config ... or somewhere else
    data_splits = ['5768', '78516', '944601']
    all_results = []

    for data_split in data_splits:
        logger.info(f"Loading dataset split: {data_split}")
        try:
            dataset = load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree", data_split, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load dataset with split {data_split}: {str(e)}")
            raise

        logger.info(f"Dataset type: {type(dataset['train'])}")
        logger.info(f"First item in dataset: {dataset['train'][0]}")

        for i in tqdm(
            range(0, len(dataset["train"]), args.batch_size), desc="Processing batches"
        ):
            batch_start_time = time.time()
            batch = dataset["train"][i : i + args.batch_size]
            
            logger.info(f"Batch type: {type(batch)}")
            logger.info(f"Batch keys: {batch.keys()}")
            logger.info(f"Batch size: {len(batch['sentence'])}")
            logger.info(f"First sentence in batch: {batch['sentence'][0] if len(batch['sentence']) > 0 else 'Empty batch'}")
            prompts = prepare_batch(batch, args)

            if not any(prompts):
                logger.warning(
                    f"All prompts in batch {i//args.batch_size + 1} failed to prepare. Skipping batch."
                )
                continue

            batch_results = []

            for idx, prompt in enumerate(prompts):
                if prompt is None:
                    continue
                try:
                    model_response = process_api_call(
                        prompt=prompt,
                        model=config["fpb"]["model_name"],
                        max_tokens=config["fpb"]["max_tokens"],
                        temperature=config["fpb"]["temperature"],
                        top_p=config["fpb"]["top_p"],
                        repetition_penalty=config["fpb"]["repetition_penalty"],
                        stop=None,
                    )

                batch_results = process_batch_response(
                    model_response, batch, "fpb", args.model
                )
                all_results.extend(batch_results)

                process_api_response(batch_results, "fpb", args.model)

                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                total_time += batch_time
                total_batches += 1
                logger.info(
                    f"Processed batch {i//args.batch_size + 1}, sentences {i+1}-{min(i+args.batch_size, len(dataset['train']))}"
                )
                logger.info(f"Batch processing time: {batch_time:.2f} seconds")

            except Exception as e:
                logger.error(f"Error processing batch {i//args.batch_size + 1}: {str(e)}")
                for dp in batch:
                    all_results.append(
                        {
                            "sentence": dp["sentence"],
                            "actual_label": dp["label"],
                            "llm_response": "error",
                            "complete_response": str(e),
                        }
                    )

            time.sleep(1)  # Rate limiting

    df = pd.DataFrame(all_results)
    logger.info(f"FPB inference completed. Total processed sentences: {len(df)}")
    if total_batches > 0:
        avg_time_per_batch = total_time / total_batches
        logger.info(f"Average time per batch: {avg_time_per_batch:.2f} seconds")

    return df
