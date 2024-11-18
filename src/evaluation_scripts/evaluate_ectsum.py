import pandas as pd
import logging
from datetime import date
from pathlib import Path
from litellm import completion 
from superflue.together_code.tokens import tokens
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def summarization_prompt(input_text: str):
    # Adjust the prompt to generate financial summaries based on the input text
    prompt = f'''Generate a financial summary in about 50 words in line-by-line bullet format based on the following input. The summary should include key financial information such as earnings per share, revenue, and other significant figures.
                
                Here is the input to analyze:
                "{input_text}"'''
    return prompt

def extract_and_evaluate_responses(args):
    together.api_key = args.api_key  # type: ignore
    results_file = (
        ROOT_DIR
        / "results"
        / args.task
        / f"{args.task}_{args.model}_{args.date}.csv"
    )

    df = pd.read_csv(results_file)
    generated_summaries = []
    # Assuming the output column contains the expected summaries
    correct_summaries = df['output'].tolist()

    for i, input_text in enumerate(df["input"]):
        try:
            model_response = completion(  # type: ignore
                messages=[{"role": "user", "content": summarization_prompt(input_text)}],
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            generated_summary = model_response.choices[0].message.content.strip()  # type: ignore
            generated_summaries.append(generated_summary)
            logger.info(f"Processed {i + 1}/{len(df)} inputs.")
        except Exception as e:
            logger.error(f"Error processing input {i}: {e}")
            generated_summaries.append(None)

    # Add generated summaries to the dataframe
    df['generated_summaries'] = generated_summaries

    # Evaluate the performance
    correct_predictions = sum(1 for x, y in zip(correct_summaries, generated_summaries) if x == y)
    total_predictions = len(correct_summaries)
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
