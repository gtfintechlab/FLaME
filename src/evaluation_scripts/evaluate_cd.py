import pandas as pd
import logging
from datetime import date
from pathlib import Path
import together


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def extraction_prompt(llm_response: str):
    
    prompt = f'''You are given the following model response. Extract the token classifications (cause, effect, or other) for each token. The classification must follow this scheme:
                - 'B-CAUSE': The beginning of a cause phrase.
                - 'I-CAUSE': A token inside a cause phrase, but not the first token.
                - 'B-EFFECT': The beginning of an effect phrase.
                - 'I-EFFECT': A token inside an effect phrase, but not the first token.
                - 'O': A token that is neither part of a cause nor an effect.
                
                Extract the classification for each token in the format 'token:label'.
                
                Model response: "{llm_response}"'''
    return prompt



def extract_and_evaluate_responses(args):
    together.api_key = args.api_key  
    results_file = (
        ROOT_DIR
        / "results"
        / args.task
        / f"{args.task}_{args.model}_{args.date}.csv"
    )

   
    df = pd.read_csv(results_file)
    extracted_tags = []  
    actual_tags = df['actual_tags'].tolist() 

    for i, llm_response in enumerate(df["complete_responses"]):
        try:
           
            model_response = together.Complete.create(
                prompt=extraction_prompt(llm_response),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )

            extracted_tag = model_response["output"]["choices"][0]["text"].strip().split()
            extracted_tags.append(extracted_tag)
            logger.info(f"Processed {i + 1}/{len(df)} responses.")
        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_tags.append(None)

   
    df['extracted_tags'] = extracted_tags

   
    correct_predictions = 0
    total_predictions = 0
    for actual, predicted in zip(actual_tags, extracted_tags):
        if predicted is not None:
            correct_predictions += sum(1 for x, y in zip(actual, predicted) if x == y)
            total_predictions += len(actual)
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0


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
