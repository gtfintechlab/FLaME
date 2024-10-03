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
from together import Together

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
    client = Together()

    for i in tqdm(range(len(dataset['test'])), desc="Processing sentences"):  # type: ignore
        sentence = dataset['test'][i]["sentence"] # type: ignore
        actual_label = dataset['test'][i]["label"] # type: ignore
        sentences.append(sentence)
        actual_labels.append(actual_label)
        try:
            logger.debug(f"Processing sentence {i+1}/{len(dataset['test'])}") # type: ignore
            model_response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": fpb_prompt(sentence, prompt_format='superflue')}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model)
            )
            logger.debug(f"Model response: {model_response}")
            complete_responses.append(model_response)
            response_label = model_response.choices[0].message.content # type: ignore
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
        / 'fpb/fpb_meta-llama-3.1-8b/'
        / f"{'fpb'}_{'llama-3.1'}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    time.sleep(10.0)
    logger.info(f"Results saved to {results_path}")

    return df
