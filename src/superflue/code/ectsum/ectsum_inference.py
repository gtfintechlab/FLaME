import time
from datetime import date
import pandas as pd
from datasets import load_dataset
from litellm import completion
from superflue.code.prompts import ectsum_prompt
from superflue.utils.logging_utils import get_logger
from superflue.utils.path_utils import get_inference_path

logger = get_logger(__name__)


def ectsum_inference(args):
    today = date.today()
    logger.info(f"Starting ECTSum inference on {today}")

    # Load the ECTSum dataset (test split)
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/ECTSum", trust_remote_code=True)
    documents = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    logger.info(f"Starting inference on ECTSum with model {args.model}...")

    # Iterate through the test split of the dataset
    for i in range(len(dataset["test"])):  # type: ignore
        document = dataset["test"][i][
            "context"
        ]  # Extract document (context) # type: ignore
        actual_label = dataset["test"][i][
            "response"
        ]  # Extract the actual label (response) # type: ignore
        documents.append(document)
        actual_labels.append(actual_label)

        try:
            logger.info(f"Processing document {i+1}/{len(dataset['test'])}")  # type: ignore
            # Generate the model's response using Together API
            model_response = completion(
                model=args.model,
                messages=[{"role": "user", "content": ectsum_prompt(document)}],
                tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )

            complete_responses.append(model_response)
            response_text = model_response.choices[0].message.content.strip()  # type: ignore
            llm_responses.append(response_text)

            logger.info(f"Model response for document {i+1}: {response_text}")

        except Exception as e:
            logger.error(f"Error processing document {i+1}: {e}")
            complete_responses.append(None)
            llm_responses.append(None)
            time.sleep(10.0)
            continue

    df = pd.DataFrame(
        {
            "documents": documents,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )
    results_path = get_inference_path(args.dataset, args.model)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)

    return df
