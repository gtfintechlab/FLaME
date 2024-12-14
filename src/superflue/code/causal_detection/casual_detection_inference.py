import time
from litellm import completion
import pandas as pd
from datasets import load_dataset
from superflue.code.prompts import causal_detection_prompt
from superflue.utils.logging_utils import get_logger
from superflue.utils.save_utils import save_inference_results

logger = get_logger(__name__)


def casual_detection_inference(args):
    dataset = load_dataset("gtfintechlab/CausalDetection", trust_remote_code=True)

    # Initialize lists to store tokens, actual tags, predicted tags, and complete responses
    tokens_list = []
    actual_tags = []
    predicted_tags = []
    complete_responses = []

    for entry in dataset["test"]:  # type: ignore
        tokens1 = entry["tokens"]  # type: ignore
        actual_tag = entry["tags"]  # type: ignore

        tokens_list.append(tokens1)
        actual_tags.append(actual_tag)

        try:
            logger.info(f"Processing entry {len(tokens_list)}")
            # Causal Detection-specific prompt logic to classify each token
            model_response = completion(
                model=args.model,
                messages=[
                    {"role": "user", "content": causal_detection_prompt(tokens1)}
                ],
                temperature=args.temperature,
                tokens=args.max_tokens,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                # stop=tokens(args.model)
            )
            complete_responses.append(model_response)
            response_label = model_response.choices[0].message.content  # type: ignore
            response_tags = (
                response_label.split()
            )  # Assumed token-wise classification # type: ignore
            predicted_tags.append(response_tags)

        except Exception as e:
            logger.error(f"Error processing entry {len(tokens_list)}: {e}")
            complete_responses.append(None)
            predicted_tags.append(None)
            time.sleep(20.0)

    # Periodically save results
    df = pd.DataFrame(
        {
            "tokens": tokens_list,
            "actual_tags": actual_tags,
            "predicted_tags": predicted_tags,
            "complete_responses": complete_responses,
        }
    )

    model_parts = args.inference_model.split("/")
    provider = model_parts[0] if len(model_parts) > 1 else "unknown"
    model_name = model_parts[-1]

    metadata = {
        "model": args.inference_model,
        "provider": provider,
        "model_name": model_name,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "repetition_penalty": args.repetition_penalty,
        "dataset_org": args.dataset_org,
        # "success_rate": success_rate,
    }

    save_inference_results(
        df=df, task="causal_detection", model=args.inference_model, metadata=metadata
    )

    return df
