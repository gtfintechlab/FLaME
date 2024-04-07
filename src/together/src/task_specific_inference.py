import pandas as pd
import time
from datasets import load_dataset
from datetime import date
import nltk
import together
from prompts_and_tokens import tokens, fomc_prompt, finentity_prompt, ectsum_prompt, fpb_prompt, numclaim_prompt
from nltk.tokenize import word_tokenize
nltk.download('punkt')


def task_inference(args, dataset_name, config_name, sentence_key, label_key, prompt_function):
    together.api_key = args.api_key
    dataset = load_dataset(dataset_name, config_name, token=args.hf_token)

    sentences, llm_responses, llm_first_word_responses, actual_labels, complete_responses = [], [], [], [], []
    start_t = time.time()
    for i, data in enumerate(dataset['test']):
        sentence = data[sentence_key]
        actual_label = data[label_key]
        sentences.append(sentence)
        actual_labels.append(actual_label)
        try:
            model_response = together.Complete.create(
                prompt=prompt_function(sentence),
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
        except Exception as e:
            print(e)
            i -= 1
            time.sleep(10.0)
    
    return pd.DataFrame({
        'sentences': sentences,
        'llm_responses': llm_responses,
        'llm_first_word_responses': llm_first_word_responses,
        'actual_labels': actual_labels,
        'complete_responses': complete_responses
    })
