import together
import time
from datetime import date
import pandas as pd
from datasets import load_dataset

from superflue.together_code.prompts import banking77_prompt
from superflue.together_code.tokens import tokens
from superflue.config import RESULTS_DIR
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL
from together import Together
from tqdm import tqdm

logger = setup_logger(
    name="banking77_inference", log_file=LOG_DIR / "banking77_inference.log", level=LOG_LEVEL
)


def banking77_inference(args):
    dataset = load_dataset("gtfintechlab/banking77", trust_remote_code=True)
    today = date.today()
    documents = []
    llm_responses = []
    actual_labels = []
    complete_responses = []
    client = Together()
    for i in tqdm(range(len(dataset["test"])), desc="Processing sentences"): # type: ignore
        document = dataset["test"][i]["text"] # type: ignore
        actual_label = dataset["test"][i]["label"] # type: ignore
        documents.append(document)
        actual_labels.append(actual_label)
        try:
            logger.debug(f"Processing sentence {i+1}/{len(dataset['test'])}") # type: ignore
            model_response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": banking77_prompt(document)}],
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
            logger.error(f"Error processing sentence {i+1}: {e}")
            complete_responses.append(None)
            llm_responses.append(None)
            time.sleep(10.0)

    df = pd.DataFrame(
        {
            "documents": documents,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )
    # results_path = (
    #     RESULTS_DIR
    #     / args.task
    #     / f"{args.task}_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    # )
    # results_path.parent.mkdir(parents=True, exist_ok=True)
    # df.to_csv(results_path, index=False)
    # logger.info(f"Inference completed for {i}. Results saved to {results_path}")
    return df
