import time
from datetime import date
from pathlib import Path
from litellm import completion 
import pandas as pd
from datasets import load_dataset
import together
from superflue.together_code.prompts import causal_detection_prompt
from superflue.together_code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="cd_inference", log_file=LOG_DIR / "cd_inference.log", level=LOG_LEVEL
)

def cd_inference(args):
    today = date.today()
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
                messages=[{"role": "user", "content": causal_detection_prompt(tokens1)}],
                temperature=args.temperature,
                tokens=args.max_tokens,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model)
            )
            complete_responses.append(model_response)
            response_label = model_response.choices[0].message.content # type: ignore
            predicted_tag = response_label.split()  # Assumed token-wise classification # type: ignore
            predicted_tags.append(predicted_tag)

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

    results_path = (
        RESULTS_DIR
        / "causal_detection"
        / f"{args.dataset}_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)

    logger.info(f"Inference completed. Results saved to {results_path}")
    return df
