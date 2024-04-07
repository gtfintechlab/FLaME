import together
# from utils.prompt_generator import fpb_prompt
import pandas as pd
import time
# from together_pipeline import generate
from datasets import load_dataset
from datetime import date
import nltk
from prompts_and_tokens import tokens, numclaim_prompt
from nltk.tokenize import word_tokenize
import together
import pandas as pd
import time
from datasets import load_dataset
from datetime import date
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')



def numclaim_inference(args):
    together.api_key = args.api_key
    today = date.today()
    
    dataset = load_dataset("gtfintechlab/Numclaim", token=args.hf_token)
    
    # Initialize lists to store actual labels and model responses
    sentences = []
    llm_responses = []
    llm_first_word_responses = []
    actual_labels = []
    complete_responses = []

    # Iterating through the train split of the dataset
    start_t = time.time()
    for i in range(len(dataset['test'])):
        sentence = dataset['test'][i]['context']
        actual_label = dataset['test'][i]['response']
        sentences.append(sentence)
        actual_labels.append(actual_label)
        try:
            model_response = together.Complete.create(prompt=numclaim_prompt(sentence),
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
            df = pd.DataFrame({'sentences': sentences, 'complete_responses': complete_responses, 'llm_responses': llm_responses, 'llm_first_word_responses': llm_first_word_responses, 'actual_labels': actual_labels})
        except Exception as e:
            print(e)
            i = i - 1
            time.sleep(10.0)
            
    return df


