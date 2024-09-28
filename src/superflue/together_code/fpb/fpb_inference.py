import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from superflue.together_code.prompts import fpb_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.config import LOG_DIR, PACKAGE_DIR
logger = setup_logger("fpb_inference", LOG_DIR / "fpb_inference.log")
import yaml

def prepare_batch(data_points: List[Dict[str, Any]], args) -> List[str]:
    prompts = []
    for dp in data_points:
        try:
            prompt = fpb_prompt(
                sentence=dp["sentence"], prompt_format=args.prompt_format
            )
            prompts.append(prompt)
        except Exception as e:
            logger.error(
                f"Error preparing prompt for sentence: {dp['sentence']}. Error: {str(e)}"
            )
            prompts.append(None)
    return prompts


def process_batch_response(
    batch_response: Dict[str, Any],
    data_points: List[Dict[str, Any]],
    task: str,
    model: str,
) -> List[Dict[str, Any]]:
    results = []
    for i, choice in enumerate(batch_response["output"]["choices"]):
        try:
            result = {
                "sentence": data_points[i]["sentence"],
                "actual_label": data_points[i]["label"],
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
                f"Error processing response for sentence: {data_points[i]['sentence']}. Error: {str(e)}"
            )
            results.append(
                {
                    "sentence": data_points[i]["sentence"],
                    "actual_label": data_points[i]["label"],
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
    data_splits = ["sentences_allagree"]

    all_results = []

    for data_split in data_splits:
        logger.info(f"Loading dataset split: {data_split}")
        try:
            dataset = load_dataset("financial_phrasebank", data_split, token=args.hf_token)
        except Exception as e:
            logger.error(f"Failed to load dataset with split {data_split}: {str(e)}")
            raise

        for i in tqdm(
            range(0, len(dataset["train"]), args.batch_size), desc="Processing batches"
        ):
            batch_start_time = time.time()
            batch = dataset["train"][i : i + args.batch_size]
            prompts = prepare_batch(batch, args)

            if not any(prompts):
                logger.warning(
                    f"All prompts in batch {i//args.batch_size + 1} failed to prepare. Skipping batch."
                )
                continue

            try:
                model_response = process_api_call(
                    prompts=[p for p in prompts if p is not None],
                    model=config["fpb"]["model_name"],
                    max_tokens=config["fpb"]["max_tokens"],
                    temperature=config["fpb"]["temperature"],
                    # top_k=config["fpb"]["top_k"],
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
