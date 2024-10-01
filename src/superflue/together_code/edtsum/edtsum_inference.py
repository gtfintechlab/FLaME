import time
import pandas as pd
from datasets import load_dataset
import together
from superflue.together_code.prompts import edtsum_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.together_code.tokens import tokens
from superflue.config import LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="edtsum_inference", log_file=LOG_DIR / "edtsum_inference.log", level=LOG_LEVEL
)


def edtsum_inference(args):
    # today = date.today()

    dataset = load_dataset("gtfintechlab/EDTSum", trust_remote_code=True)

    documents = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    # start_t = time.time()
    for i in range(len(dataset["test"])): # type: ignore
        document = dataset["test"][i]["text"] # type: ignore
        actual_label = dataset["test"][i]["answer"] # type: ignore
        documents.append(document)
        actual_labels.append(actual_label)

        try:
            model_response = together.Complete.create(
                prompt=edtsum_prompt(document),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model)
            )
            complete_responses.append(model_response)

            response_label = model_response["choices"][0]["text"]
            # print(response_label)
            llm_responses.append(response_label)

        except Exception as e:
            logger.error(f"Error at index {i}: {e}")
            complete_responses.append(None)
            llm_responses.append(None)
            time.sleep(30.0)

    assert (
        len(documents)
        == len(llm_responses)
        == len(actual_labels)
        == len(complete_responses)
    ), "Lists are not of equal length!"

    df = pd.DataFrame(
        {
            "documents": documents,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    return df
