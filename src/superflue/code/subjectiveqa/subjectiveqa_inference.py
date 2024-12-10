import time
from datetime import date
import os
import pandas as pd
from datasets import load_dataset

from litellm import completion 
from superflue.code.prompts import subjectiveqa_prompt
from superflue.code.tokens import tokens

from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="subjectiveqa_inference",
    log_file=LOG_DIR / "subjectiveqa_inference.log",
    level=LOG_LEVEL,
)

def subjectiveqa_inference(args):
    
    definition_map = {
        "RELEVANT": "The speaker has answered the question entirely and appropriately.",
        "SPECIFIC": "The speaker includes specific and technical details in the answer.",
        "CAUTIOUS": "The speaker answers using a more conservative, risk-averse approach.",
        "ASSERTIVE": "The speaker answers with certainty about the company's events.",
        "CLEAR": "The speaker is transparent in the answer and about the message to be conveyed.",
        "OPTIMISTIC": "The speaker answers with a positive tone regarding outcomes.",
    }
    
    today = date.today()
    logger.info(f"Starting SubjectiveQA inference on {today}")

    # Load only the 'test' split of the dataset
    dataset = load_dataset("gtfintechlab/subjectiveqa", "5768", split="test", trust_remote_code=True)

    # Initialize lists to store actual labels, model responses, and complete responses
    questions = []
    answers = []
    feature_responses = {feature: [] for feature in definition_map.keys()}
    feature_labels = {feature: [] for feature in definition_map.keys()}
    complete_responses = []  # To store complete responses for further analysis

    logger.info(f"Starting inference on SubjectiveQA with model {args.model}...")

    for i, row in enumerate(dataset):
        logger.info(f"Processing row {i+1}/{len(dataset)}") # type: ignore
        question = row["QUESTION"]
        answer = row["ANSWER"]
        questions.append(question)
        answers.append(answer)
        
        row_data = {"questions": [question], "answers": [answer]}  # Collect row-specific data
        
        try:
            for feature, definition in definition_map.items():
                actual_label = row[feature]
                feature_labels[feature].append(actual_label)
                row_data[f"{feature}_actual_label"] = [actual_label]
                
                try:
                    model_response = completion(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": "You are an expert sentence classifier."},
                            {"role": "user", "content": subjectiveqa_prompt(feature, definition, question, answer)},
                        ],
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        stop=tokens(args.model),
                    )
                    
                    response_label = model_response.choices[0].message.content.strip()  # type: ignore
                    feature_responses[feature].append(response_label)
                    row_data[feature] = [response_label]
                    complete_responses.append(model_response)  # Store the full response

                    logger.info(f"Processed {feature} for row {i+1}: {response_label}")

                except Exception as e:
                    logger.error(f"Error processing {feature} for row {i+1}: {e}")
                    feature_responses[feature].append("error")
                    row_data[feature] = ["error"]

        except Exception as e:
            logger.error(f"Error processing row {i+1}: {e}")
            for feature in definition_map.keys():
                feature_responses[feature].append("error")
            complete_responses.append(None)
            time.sleep(10.0)
            continue

    # Create a DataFrame to store the results
    df = pd.DataFrame(
        {
            "questions": questions,
            "answers": answers,
            **{f"{feature}_response": feature_responses[feature] for feature in definition_map.keys()},
            **{f"{feature}_actual_label": feature_labels[feature] for feature in definition_map.keys()}
            # "complete_responses": complete_responses,  # Add complete responses for each question
        }
    )

    # Define new path to save results (e.g., CSV file inside subjectiveqa folder)
    results_path = (
        RESULTS_DIR
        / "subjectiveqa"
        / f"subjectiveqa_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df
