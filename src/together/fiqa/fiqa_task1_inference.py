import time
import pandas as pd
from datasets import load_dataset
import together
from src.together.prompts import fiqa_prompt
from src.together.tokens import tokens

def fiqa_inference(args):
    together.api_key = args.api_key
    
   
    dataset = load_dataset("gtfintechlab/FiQA_Task1", split="test", token=args.hf_token)
    
    context = []
    llm_responses = []
    actual_targets = []
    actual_sentiments = []
    complete_responses = []
    
    start_time = time.time()
    

    for entry in dataset:
        sentence = entry['sentence']  
        snippets = entry['snippets']  
        target = entry['target'] 
        sentiment_score = entry['sentiment_score'] 
        
       
        combined_text = f"Sentence: {sentence}. Snippets: {snippets}. Target aspect: {target}. What is the sentiment?"
        context.append(combined_text)
        
        actual_targets.append(target)
        actual_sentiments.append(sentiment_score)
        
        try:
           
            model_response = together.Complete.create(
                prompt=fiqa_prompt(combined_text),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            
            complete_responses.append(model_response)
            response_label = model_response["output"]["choices"][0]["text"]
            llm_responses.append(response_label)
            
            print(response_label)
            df = pd.DataFrame(
                {
                    "context": context,
                    "llm_responses": llm_responses,
                    "actual_target": actual_targets,
                    "actual_sentiment": actual_sentiments,
                    "complete_responses": complete_responses,
                }
            )
            time.sleep(10) 

        except Exception as e:
            print(f"Error encountered: {e}")
            time.sleep(20.0)  
    
    return df

