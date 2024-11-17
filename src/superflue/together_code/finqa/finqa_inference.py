import time

import pandas as pd
from datasets import load_dataset

from litellm import completion 
from superflue.together_code.prompts import finqa_prompt
from superflue.together_code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import LOG_DIR, LOG_LEVEL

# TODO: (Glenn) Is FinQA saving results to a file properly?

logger = setup_logger(
    name="finqa_inference", log_file=LOG_DIR / "finqa_inference.log", level=LOG_LEVEL
)


def finqa_inference(args):
    
    # today = date.today()
    dataset = load_dataset("gtfintechlab/finqa", trust_remote_code=True)
    context = []
    llm_responses = []
    actual_labels = []
    complete_responses = []
    # start_t = time.time()
    for entry in dataset["test"]:  # type: ignore
        pre_text = " ".join(entry["pre_text"])  # type: ignore
        post_text = " ".join(entry["post_text"])  # type: ignore
        table_text = " ".join([" ".join(row) for row in entry["table_ori"]])  # type: ignore
        combined_text = f"{pre_text} {post_text} {table_text} {entry['question']}"  # type: ignore
        context.append(combined_text)
        actual_label = entry["answer"]  # type: ignore
        actual_labels.append(actual_label)
        try:
            model_response = completion(
                prompt=finqa_prompt(combined_text),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            complete_responses.append(model_response)
            response_label = model_response.choices[0].message.content  # type: ignore
            llm_responses.append(response_label)

            df = pd.DataFrame(
                {
                    "context": context,
                    "response": llm_responses,
                    "actual_label": actual_labels,
                    "complete_responses": complete_responses,
                }
            )
            time.sleep(10)

        except Exception as e:
            logger.error(e)
            # i = i - 1
            time.sleep(20.0)

    return df
