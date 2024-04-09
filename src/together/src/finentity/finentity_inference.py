import together
import pandas as pd
import time
from datasets import load_dataset
from datetime import date
from prompts_and_tokens import tokens, finentity_prompt


def finentity_inference(args):
    together.api_key = args.api_key
    today = date.today()
    configs = ["5768"]
    
    for config in configs:
        dataset = load_dataset("gtfintechlab/finentity", config, token=args.hf_token)

        sentences = []
        llm_responses = []
        actual_labels = []
        complete_responses = []

        for i in range(len(dataset['test'])):
            sentence = dataset['test'][i]['content']
            actual_label = dataset['test'][i]['annotations']  
            sentences.append(sentence)
            actual_labels.append(actual_label)
            success = False

            while not success:
                try:
                    model_response = together.Complete.create(
                        prompt=finentity_prompt(sentence),
                        model=args.model,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        stop=tokens(args.model)
                    )
                    success = True  
                    response_label = model_response["output"]["choices"][0]["text"]
                    llm_responses.append(response_label)
                    print(llm_responses)
                    complete_responses.append(model_response)  
                    
                except Exception as e:
                    print(e)
                    time.sleep(10.0)  

            time.sleep(10.0)

        df = pd.DataFrame({
            'sentences': sentences,
            'llm_responses': llm_responses,
            'actual_labels': actual_labels,
            'complete_responses': complete_responses 
        })
        
        

    return df
