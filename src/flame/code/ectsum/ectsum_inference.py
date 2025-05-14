from datetime import date
import pandas as pd
from datasets import load_dataset
import litellm

from litellm import completion
from flame.code.prompts_zeroshot import ectsum_zeroshot_prompt
from flame.code.prompts_fewshot import ectsum_fewshot_prompt

# from flame.code.tokens import tokens
from flame.utils.logging_utils import setup_logger
from flame.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

# Setup logger for ectsum inference
logger = setup_logger(
    name="ectsum_inference", log_file=LOG_DIR / "ectsum_inference.log", level=LOG_LEVEL
)

litellm.drop_params = True


def ectsum_inference(args):
    today = date.today()
    logger.info(f"Starting ECTSum inference on {today}")

    # Load the ECTSum dataset (test split)
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/ECTSum", trust_remote_code=True)

    results_path = (
        RESULTS_DIR
        / "ectsum"
        / f"ectsum_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize lists to store documents, actual labels, model responses, and complete responses
    documents = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    if args.prompt_format == "fewshot":
        ectsum_prompt = ectsum_fewshot_prompt
    elif args.prompt_format == "zeroshot":
        ectsum_prompt = ectsum_zeroshot_prompt

    logger.info(f"Starting inference on ECTSum with model {args.model}...")

    # Iterate through the test split of the dataset
    for i in range(len(dataset["test"])):  # type: ignore
        document = dataset["test"][i][
            "context"
        ]  # Extract document (context) # type: ignore
        actual_label = dataset["test"][i][
            "response"
        ]  # Extract the actual label (response) # type: ignore
        # documents.append(document)
        # actual_labels.append(actual_label)

        try:
            logger.info(f"Processing document {i + 1}/{len(dataset['test'])}")  # type: ignore
            # Generate the model's response using Together API
            model_response = completion(
                model=args.model,
                messages=[{"role": "user", "content": ectsum_prompt(document)}],
                # tokens=args.max_tokens,
                temperature=args.temperature,
                # top_k=args.top_k,
                # top_p=args.top_p,
                # repetition_penalty=args.repetition_penalty,
                # stop=tokens(args.model),
            )

            # Append the model response and complete response for the document
            # complete_responses.append(model_response)
            response_text = model_response.choices[0].message.content.strip()  # type: ignore
            # llm_responses.append(response_text)

            logger.info(f"Model response for document {i + 1}: {response_text}")

        except Exception as e:
            # Log the error and retry the same document after a delay
            logger.error(f"Error processing document {i + 1}: {e}")
            # documents.append(document if 'document' in locals() else None)
            # actual_labels.append(actual_label if 'actual_label' in locals() else None)
            # complete_responses.append("Error")
            # llm_responses.append("Error")
            response_text = "Error"
            model_response = "Error"

        documents.append(document)
        actual_labels.append(actual_label)
        llm_responses.append(response_text)
        complete_responses.append(model_response)

    # Create the final DataFrame after the loop
    df = pd.DataFrame(
        {
            "documents": documents,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    logger.info(f"Inference completed. Returning DataFrame with {len(df)} rows.")

    return df
