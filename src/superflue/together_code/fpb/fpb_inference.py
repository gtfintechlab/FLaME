import together
import pandas as pd
import time
from datasets import load_dataset
from datetime import date
from superflue.together_code.prompts import fpb_prompt
from superflue.together_code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from tqdm import tqdm
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="fpb_inference", log_file=LOG_DIR / "fpb_inference.log", level=LOG_LEVEL
)


today = date.today()


def fpb_inference(args):
    together.api_key = args.api_key
    # TODO: (Glenn) Very low priority, we can set the data_split as configurable in yaml
    # data_splits = ["sentences_50agree", "sentences_66agree", "sentences_75agree", "sentences_allagree"]
    data_splits = ["sentences_allagree"]
    for data_split in data_splits:
        dataset = load_dataset("financial_phrasebank", data_split)

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
                    RESULTS_DIR
                    / args.task
                    / f"{args.task}_{args.model_name}_{date.today().strftime('%d_%m_%Y')}.csv"
                )
                results_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(results_path, index=False)
                time.sleep(10.0)  # TODO: (Glenn) Determine if this sleep is necessary
                logger.info(f"Results saved to {results_path}")

    return df
