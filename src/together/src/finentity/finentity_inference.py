import together
# from utils.prompt_generator import fpb_prompt
import pandas as pd
import time
from prompts_and_tokens import finentity_prompt, tokens
# from together_pipeline import generate
from datasets import load_dataset
from datetime import date
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


def finentity_inference(args):
    together.api_key = args.api_key
    today = date.today()
    # OPTIONAL TODO: make configs an argument of some kind LOW LOW LOW PRIORITY
    #configs = ["5768", "78516", "944601"]
    configs = ["5768"]
    
    for config in configs:
        dataset = load_dataset("gtfintechlab/finentity", config, token=args.hf_token)

        # Initialize lists to store actual labels and model responses
    sentences = []
    llm_responses = []
    llm_first_word_responses = []
    actual_labels = []
    complete_responses = []

        # Iterating through the train split of the dataset
    start_t = time.time()
    for i in range(len(dataset['test'])):
        sentence = dataset['test'][i]['content']
        actual_label = dataset['train'][i]['annotations']
        sentences.append(sentence)
        actual_labels.append(actual_label)
        try:
            model_response = together.Complete.create(prompt=finentity_prompt(sentence),
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
            words = word_tokenize(response_label.strip())
            llm_first_word_responses.append(words[0])
            llm_responses.append(response_label)
            df = pd.DataFrame({'sentences': sentences, 'llm_responses': llm_responses, 'llm_first_word_responses': llm_first_word_responses, 'actual_labels': actual_labels, 'complete_responses': complete_responses})
        except Exception as e:
            print(e)
            i = i - 1
            time.sleep(10.0)
            
    return df

