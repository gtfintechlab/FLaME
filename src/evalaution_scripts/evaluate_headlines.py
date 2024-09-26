import pandas as pd
import logging
from datetime import date
from pathlib import Path
import together

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def extraction_prompt(llm_response: str):
    prompt = f'''Based on the following attributes, give a score of 0 if the news headline does not contain the information and 1 if it does:
                    - Price or Not: Does the news item talk about price or not?
                    - Direction Up: Does the news headline talk about price going up or not?
                    - Direction Down: Does the news headline talk about price going down or not?
                    - Direction Constant: Does the news headline talk about price remaining constant or not?
                    - Past Price: Does the news headline talk about an event in the past?
                    - Future Price: Does the news headline talk about an event in the future?
                    - Past News: Does the news headline talk about a general event (apart from prices) in the past?
                Extract the relevant information from the following response and provide a score of 0 or 1 for each attribute:
                "{llm_response}"'''

    return prompt


def extract_and_evaluate_responses(args):
    together.api_key = args.api_key # type: ignore
    results_file = (
        ROOT_DIR
        / "results"
        / args.task
        / f"{args.task}_{args.model}_{args.date}.csv"
    )

    # Load the CSV file with the LLM responses
    df = pd.read_csv(results_file)
    extracted_labels = []
    correct_labels = df['actual_labels'].tolist()

    for i, llm_response in enumerate(df["llm_responses"]):
        try:
            model_response = together.Complete.create( # type: ignore
                prompt=extraction_prompt(llm_response),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            extracted_label = model_response["output"]["choices"][0]["text"].strip() # type: ignore
            extracted_labels.append(extracted_label)
            logger.info(f"Processed {i + 1}/{len(df)} responses.")
        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_labels.append(None)

    # Add extracted labels to the dataframe
    df['extracted_labels'] = extracted_labels

    # Evaluate the performance
    correct_predictions = sum(1 for x, y in zip(correct_labels, extracted_labels) if x == y)
    total_predictions = len(correct_labels)
    accuracy = correct_predictions / total_predictions

    # Save the evaluation results
    evaluation_results_path = (
        ROOT_DIR
        / "evaluation_results"
        / args.task
        / f"evaluation_{args.task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    df.to_csv(evaluation_results_path, index=False)

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}. Results saved to {evaluation_results_path}")
    return df, accuracy

tokens_map = {"meta-llama/Llama-2-7b-chat-hf": ["<human>", "\n\n"]}
def tokens(model_name):
    return tokens_map.get(model_name, [])


