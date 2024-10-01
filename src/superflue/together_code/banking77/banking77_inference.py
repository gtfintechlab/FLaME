import together
import time
from datetime import date
import pandas as pd
from datasets import load_dataset

from superflue.together_code.prompts import banking77_prompt
from superflue.together_code.chat import get_stop_tokens
from superflue.config import RESULTS_DIR
from superflue.utils.logging_utils import setup_logger
from superflue.config import LOG_LEVEL

logger = setup_logger(
    name="banking77_inference", log_file="banking77_inference.log", level=LOG_LEVEL
)


def banking77_inference(args):
    dataset = load_dataset("gtfintechlab/banking77", trust_remote_code=True)
    today = date.today()
    documents = []
    llm_responses = []
    actual_labels = []
    complete_responses = []
    for i in range(len(dataset["test"])): # type: ignore
        document = dataset["test"][i]["text"] # type: ignore
        actual_label = dataset["test"][i]["label"] # type: ignore
        documents.append(document)
        actual_labels.append(actual_label)
        try:
            model_response = together.Complete.create(
                prompt=banking77_prompt(document),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=get_stop_tokens(args.model),
            )
            complete_responses.append(model_response)
            response_label = model_response["output"]["choices"][0]["text"]
            llm_responses.append(response_label)
            df = pd.DataFrame(
                {
                    "documents": documents,
                    "llm_responses": llm_responses,
                    "actual_labels": actual_labels,
                    "complete_responses": complete_responses,
                }
            )
            results_path = (
                RESULTS_DIR
                / args.task
                / f"{args.task}_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
            )
            results_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(results_path, index=False)
            logger.info(f"Inference completed for {i}. Results saved to {results_path}")
        except Exception as e:
            print(e)
            i = i - 1
            documents.pop()
            actual_labels.pop()

            time.sleep(20.0)

        return df
