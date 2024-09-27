import time
import pandas as pd
from datasets import load_dataset
import together
from src.together.prompts import fiqa_task2_prompt  
from src.together.tokens import tokens

def fiqa_task2_inference(args):
    together.api_key = args.api_key
    

    dataset = load_dataset("gtfintechlab/FiQA_Task2", split="test", token=args.hf_token)
    
    context = []
    llm_responses = []
    actual_answers = []
    complete_responses = []
    
    start_time = time.time()
    
    
    for entry in dataset:
        question = entry['question']  
        actual_answer = entry['answer']  

        combined_text = f"Question: {question}. Provide a concise and accurate answer."
        context.append(combined_text)
        
       
        actual_answers.append(actual_answer)
        
        try:

            model_response = together.Complete.create(
                prompt=fiqa_task2_prompt(combined_text), 
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                # stop=tokens(args.model),  
            )
            
            complete_responses.append(model_response)
            response_label = model_response["output"]["choices"][0]["text"]
            llm_responses.append(response_label)
           # print(llm_responses)
            
           
            # print(f"Model response for '{question}': '{complete_responses}'")
            
            
            df = pd.DataFrame(
                {
                    "question": context,
                    "llm_responses": llm_responses,
                    "actual_answers": actual_answers,
                    "complete_responses": complete_responses,
                }
            )
            time.sleep(10) 

        except Exception as e:
            print(f"Error encountered: {e}")
            time.sleep(20.0) 
    
    return df