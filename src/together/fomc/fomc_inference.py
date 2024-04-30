import together
import pandas as pd
import time
from datasets import load_dataset
from datetime import date
from prompts_and_tokens import tokens, fomc_prompt
from FinGT import ROOT_DIR


def fomc_inference(args):
    together.api_key = args.api_key
    today = date.today()
    # OPTIONAL TODO: make configs an argument of some kind LOW LOW LOW PRIORITY
    # configs = ["sentences_50agree", "sentences_66agree", "sentences_75agree", "sentences_allagree"]
    dataset = load_dataset("gtfintechlab/fomc_communication", token=args.hf_token)

        # Initialize lists to store actual labels and model responses
    sentences = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

        # Iterating through the train split of the dataset
    start_t = time.time()
    for i in range(len(dataset['test'])):
        sentence = dataset['test'][i]['sentence']
        actual_label = dataset['test'][i]['label']
        sentences.append(sentence)
        actual_labels.append(actual_label)
        try:
            model_response = together.Complete.create(prompt=fomc_prompt(sentence),
                            model=args.model,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            repetition_penalty=args.repetition_penalty,
                            stop=tokens(args.model)
                            )
            complete_responses.append(model_response)
            response_label = model_response["output"]["choices"][0]["text"]
            llm_responses.append(response_label)
            df = pd.DataFrame({'sentences': sentences, 'llm_responses': llm_responses, 'actual_labels': actual_labels, 'complete_responses': complete_responses})
        except Exception as e:
            print(e)
            i = i - 1
            time.sleep(20.0)
            
        results_path = ROOT_DIR / 'results' / args.task / args.model / f"{args.task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False) 
        
    return df