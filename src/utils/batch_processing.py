from typing import List, Dict, Any

from src.together.prompts import fpb_prompt

import logging


logger = logging.getLogger(__name__)


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
