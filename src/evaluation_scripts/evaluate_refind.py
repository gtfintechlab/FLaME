import pandas as pd
import logging
from datetime import date
from pathlib import Path
import together
from together import Together
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def extraction_prompt(llm_response: str):
    prompt = f'''Extract the classification label from the following LLM response. The label should be one of the following: ‘PER-TITLE’, ‘PER-GOV’, ‘PER-ORG’, ‘PER-UNIV’, ‘ORG-ORG’, ‘ORG-MONEY’, ‘ORG-GPE’, ‘ORG-DATE’, or ‘NO-REL’. List ‘NO-REL’ if the LLM did not output a clear answer.
                
                Here is the LLM response to analyze:
                "{llm_response}"
                Provide only the label that best matches the response. Only output alphanumeric characters and spaces. Do not include any special characters or punctuation.'''

    return prompt

def extract_and_evaluate_responses(args):
    client = Together()
    together.api_key = '9c813c6191dc53f8db8a6a778744c6fb43b97eb5576b112eb6969250cd7cfb88'
    # together.api_key = args.api_key # type: ignore
    
    results_file = (
        ROOT_DIR
        / "results"
        / 'refind'
        / 'refind_meta-llama'
        / "Meta-Llama-3.1-8B-Instruct-Turbo_01_11_2024.csv"
    )

    # Load the CSV file with the LLM responses
    df = pd.read_csv(results_file)
    extracted_labels = []
    correct_labels = df['actual_labels'].tolist()

    for i, llm_response in tqdm(enumerate(df["llm_responses"])):
        try:
            model_response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",#args.model,
                messages=[{"role": "user", "content": extraction_prompt(llm_response)}],
                max_tokens=10,#args.max_tokens,
                temperature=0.0,#args.temperature,
                # top_k=args.top_k,
                top_p=0.9,#args.top_p,
                repetition_penalty=1.0,#args.repetition_penalty,
                stop=tokens("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")#args.model)
            )
            extracted_label = model_response.choices[0].message.content.strip() # type: ignore
            if (extracted_label == None):
                extracted_label = 'ERROR'
                print(f"Error processing response {i}: {llm_response}")
                logger.error(f"Error processing response {i}: {llm_response}")
            extracted_labels.append(extracted_label)
            logger.debug(f"Processed {i + 1}/{len(df)} responses.")
        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_labels.append('ERROR')

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
        / 'refind'
        / f"evaluation_{'refind'}_{'meta-llama-3.1-8b'}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(evaluation_results_path, index=False)

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}. Results saved to {evaluation_results_path}")
    accuracy = accuracy_score(correct_labels, extracted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(correct_labels, extracted_labels, average='weighted')
    eval_df = pd.DataFrame({'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1': [f1]})
    eval_df.to_csv(Path(f"{str(evaluation_results_path)[:-4]}_statistics.csv"), index=False)
    
    return df, accuracy

tokens_map = {"meta-llama/Llama-2-7b-chat-hf": ["<human>", "\n\n"]}
def tokens(model_name):
    return tokens_map.get(model_name, [])

if __name__ == "__main__":
    extract_and_evaluate_responses(None)