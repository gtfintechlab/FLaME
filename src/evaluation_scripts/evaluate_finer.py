import pandas as pd
import logging
from datetime import date
from pathlib import Path
from litellm import completion 
from superflue.together_code.tokens import tokens
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def normalize_token(token: str) -> str:
    """Normalize tokens for comparison (e.g., by converting to lowercase)."""
    return token.lower().strip()

def extraction_prompt_finer(llm_response: str):
    prompt = f'''Extract the named entity labels from the following LLM response. For each token, 
                map the labels to the following numeric values:
                    - "O" (Other): 0
                    - "PER_B" (Person_B): 1
                    - "PER_I" (Person_I): 2
                    - "LOC_B" (Location_B): 3
                    - "LOC_I" (Location_I): 4
                    - "ORG_B" (Organisation_B): 5
                    - "ORG_I" (Organisation_I): 6

                Format the output as a JSON array, where each token is represented as an object with 
                "token" and "label" fields, like this:
                [
                    {{"token": "token1", "label": label1}},
                    {{"token": "token2", "label": label2}},
                    ...
                ]

                Here is the LLM response to analyze:
                "{llm_response}"'''

    return prompt

def save_intermediate_results(df, extracted_labels, path):
    """Save intermediate results to avoid data loss."""
    df["extracted_labels"] = extracted_labels
    df.to_csv(path, index=False)
    logger.info(f"Intermediate results saved to {path}")

def extract_and_evaluate_responses(args, save_interval=10):
    results_file = (
        ROOT_DIR
        / "results"
        / args.task
        / f"{args.task}_{args.model}_{args.date}.csv"
    )

    df = pd.read_csv(results_file)
    
    extracted_labels = []
    correct_labels = df['gold_label'].tolist()
    gold_tokens = df['gold_token'].tolist()

    evaluation_results_path = (
        ROOT_DIR
        / "evaluation_results"
        / args.task
        / f"evaluation_{args.task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )

    for i, llm_response in enumerate(df["llm_responses"]):
        try:
            model_response = completion(  # type: ignore
                prompt=extraction_prompt_finer(llm_response),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            
            
            extracted_json = model_response.choices[0].message.content.strip()  # type: ignore
            extracted_tokens = json.loads(extracted_json)

            for gold_token, correct_label in zip(gold_tokens, correct_labels):
                normalized_gold_token = normalize_token(gold_token)
                matched = False
                for token_info in extracted_tokens:
                    normalized_extracted_token = normalize_token(token_info["token"])
                    if normalized_gold_token == normalized_extracted_token:
                        extracted_labels.append(token_info["label"])
                        matched = True
                        break
                if not matched:
                    logger.warning(f"No match found for gold token '{gold_token}' in response tokens.")
                    extracted_labels.append(None)
            
            logger.info(f"Processed {i + 1}/{len(df)} responses.")
            
            # Save intermediate results at the specified interval
            if (i + 1) % save_interval == 0:
                save_intermediate_results(df, extracted_labels, evaluation_results_path)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_labels.append(None)

    save_intermediate_results(df, extracted_labels, evaluation_results_path)

    # Evaluate the performance
    correct_predictions = sum(1 for x, y in zip(correct_labels, extracted_labels) if x == y)
    total_predictions = len(correct_labels)
    accuracy = correct_predictions / total_predictions

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}. Final results saved to {evaluation_results_path}")
    return df, accuracy