import pandas as pd
import logging
from datetime import date
from pathlib import Path
import together


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def extraction_prompt(llm_response: str):
   
    prompt = f'''Extract the sentiment score from the following response. The extraction should return a numeric value between -1 and 1, where -1 indicates very negative sentiment, 0 indicates neutral sentiment, and 1 indicates very positive sentiment.

    Here is the response to analyze:
    "{llm_response}"'''
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
    extracted_labels = []  
    correct_labels = df['actual_sentiment'].tolist() 

    for i, llm_response in enumerate(df["llm_responses"]):
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
            
            extracted_label = model_response["output"]["choices"][0]["text"].strip()  
            extracted_labels.append(extracted_label)
            logger.info(f"Processed {i + 1}/{len(df)} responses.")
        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_labels.append(None)

 
    df['extracted_labels'] = extracted_labels


    correct_predictions = sum(1 for x, y in zip(correct_labels, extracted_labels) if x == y)
    total_predictions = len(correct_labels)
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
