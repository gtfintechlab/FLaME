import pandas as pd
import logging
from datetime import date
from pathlib import Path
import together

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def extraction_prompt(finred_text: str):
    prompt = f'''Extract the relevant entities and their relationships from the following text. 
    Format your output as a list of tuples, where each tuple contains:
    (entity_1, entity_2, relationship).
    
    Here is the text to analyze:
    "{finred_text}"'''
    return prompt

def extract_and_evaluate_responses(args):
    together.api_key = args.api_key  # type: ignore
    results_file = (
        ROOT_DIR
        / "results"
        / args.task
        / f"{args.task}_{args.model}_{args.date}.txt"
    )

    with open(results_file, 'r') as file:
        finred_texts = file.readlines()

    generated_relationships = []
    # Assuming the last part of each line contains the expected relationships
    correct_relationships = [line.split('|')[1].strip() for line in finred_texts]

    for i, finred_text in enumerate(finred_texts):
        try:
            model_response = together.Complete.create(  # type: ignore
                prompt=extraction_prompt(finred_text),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            generated_relationship = model_response["output"]["choices"][0]["text"].strip()  # type: ignore
            generated_relationships.append(generated_relationship)
            logger.info(f"Processed {i + 1}/{len(finred_texts)} entries.")
        except Exception as e:
            logger.error(f"Error processing entry {i}: {e}")
            generated_relationships.append(None)

    # Evaluate the performance
    correct_predictions = sum(1 for x, y in zip(correct_relationships, generated_relationships) if x == y)
    total_predictions = len(correct_relationships)
    accuracy = correct_predictions / total_predictions

    # Save the evaluation results
    evaluation_results_path = (
        ROOT_DIR
        / "evaluation_results"
        / args.task
        / f"evaluation_{args.task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    pd.DataFrame({
        'input': finred_texts,
        'generated_relationships': generated_relationships,
        'correct_relationships': correct_relationships
    }).to_csv(evaluation_results_path, index=False)

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}. Results saved to {evaluation_results_path}")
    return generated_relationships, accuracy

tokens_map = {"meta-llama/Llama-2-7b-chat-hf": ["<human>", "\n\n"]}

def tokens(model_name):
    return tokens_map.get(model_name, [])