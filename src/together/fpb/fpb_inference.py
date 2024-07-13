import together
import pandas as pd
import time
from datasets import load_dataset
from datetime import date
from src.together.prompts import fpb_prompt
from pathlib import Path
from src.together.models import get_model_name

import logging
from src.together.tokens import tokens

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
today = date.today()


def fpb_inference(args):
    together.api_key = args.api_key
    # configs = ["sentences_50agree", "sentences_66agree", "sentences_75agree", "sentences_allagree"]
    configs = ["sentences_allagree"]
    for config in configs:
        dataset = load_dataset("financial_phrasebank", config, token=args.hf_token)

        sentences = []
        llm_responses = []
        actual_labels = []
        complete_responses = []

        for data_point in tqdm(dataset["train"], desc="Processing sentences"):  # type: ignore
            sentences.append(data_point["sentence"])  # type: ignore
            actual_label = data_point["label"]  # type: ignore
            actual_labels.append(actual_label)
            success = False
            while not success:
                try:
                    model_response = together.Complete.create(
                        prompt=fpb_prompt(
                            sentence=data_point["sentence"],  # type: ignore
                            prompt_format=args.prompt_format,
                        ),
                        model=args.model,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        stop=tokens(args.model),
                    )
                    success = True
                except Exception as e:
                    logger.error(f"Error: {e}. Retrying in 10 seconds.")
                    time.sleep(10.0)

                complete_responses.append(model_response)
                if "output" in model_response and "choices" in model_response["output"]:
                    response_label = model_response["output"]["choices"][0]["text"]
                    logger.debug(response_label)
                else:
                    response_label = "default_value"
                llm_responses.append(response_label)
                df = pd.DataFrame(
                    {
                        "sentences": sentences,
                        "llm_responses": llm_responses,
                        "actual_labels": actual_labels,
                        "complete_responses": complete_responses,
                    }
                )
                results_path = (
                    ROOT_DIR
                    / "results"
                    / args.task
                    / f"{args.task}_{get_model_name(args.model)}_{date.today().strftime('%d_%m_%Y')}.csv"
                )
                results_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(results_path, index=False)
                time.sleep(
                    10.0
                )  # Glenn: @Huzaifa, what is the purpose of this sleep? is it to prevent the API from being overloaded?
                logger.info(f"Results saved to {results_path}")

    return df
