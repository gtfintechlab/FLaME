import together
import pandas as pd
import time
from tqdm import tqdm
from datasets import load_dataset
from datetime import date
from superflue.together_code.prompts import fpb_prompt
from superflue.together_code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="fpb_inference", log_file=LOG_DIR / "fpb_inference.log", level=LOG_LEVEL
)

data_seed = '5768'

def fpb_inference(args):
    # TODO: (Glenn) Very low priority, we can set the data_split as configurable in yaml
    # data_splits = ["sentences_50agree", "sentences_66agree", "sentences_75agree", "sentences_allagree"]
    logger.info("Starting FPB inference")
    logger.info("Loading dataset...")
    # for data_split in data_splits:
    dataset = load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree", data_seed, trust_remote_code=True)

    sentences = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    for i in tqdm(range(len(dataset['test'])), desc="Processing sentences"):  # type: ignore
        sentence = dataset['test'][i]["sentence"] # type: ignore
        actual_label = dataset['test'][i]["label"] # type: ignore
        sentences.append(sentence)
        actual_labels.append(actual_label)
        try:
            logger.debug(f"Processing sentence {i+1}/{len(dataset['test'])}") # type: ignore
            model_response = together.Complete.create(
                prompt=fpb_prompt(
                    sentence=sentence,  # type: ignore
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

            complete_responses.append(model_response)
            if "output" in model_response and "choices" in model_response["output"]:
                response_label = model_response["output"]["choices"][0]["text"]
                logger.debug(response_label)
            else:
                response_label = "default_value"
            llm_responses.append(response_label)

        except Exception as e:
            logger.error(f"Error: {e}. Retrying in 10 seconds.")
            time.sleep(10.0)
            continue

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
    time.sleep(10.0)
    logger.info(f"Results saved to {results_path}")

    return df
