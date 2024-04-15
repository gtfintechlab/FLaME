import together
import pandas as pd
import time
from datasets import load_dataset
from datetime import date
from prompts_and_tokens import tokens, fpb_prompt

today = date.today()

def fpb_inference(args):
    together.api_key = args.api_key
    # today = date.today()
    # OPTIONAL TODO: make configs an argument of some kind LOW LOW LOW PRIORITY
    # configs = ["sentences_50agree", "sentences_66agree", "sentences_75agree", "sentences_allagree"]
    configs = ["sentences_allagree"]
    for config in configs:
        dataset = load_dataset("financial_phrasebank", config, token=args.hf_token)

        # Initialize lists to store actual labels and model responses
        sentences = []
        llm_responses = []
        actual_labels = []
        complete_responses = []

        # Iterating through the train split of the dataset
        for data_point in dataset['train']:
            sentences.append(data_point['sentence'])
            actual_label = data_point['label']
            actual_labels.append(actual_label)
            success = False
            while not success:
                try:
                    model_response = together.Complete.create(prompt=fpb_prompt(sentence=data_point['sentence'], prompt_format=args.prompt_format),
                                model=args.model,
                                max_tokens=args.max_tokens,
                                temperature=args.temperature,
                                top_k=args.top_k,
                                top_p=args.top_p,
                                repetition_penalty=args.repetition_penalty,
                                stop=tokens(args.model)
                                )
                    success = True
                except Exception as e:
                    print(e)
                    time.sleep(10.0)

                complete_responses.append(model_response)
                response_label = model_response["output"]["choices"][0]["text"]
                print(response_label)
                llm_responses.append(response_label)
                df = pd.DataFrame({'sentences': sentences, 'llm_responses': llm_responses, 'actual_labels': actual_labels, 'complete_responses': complete_responses})
                df.to_csv('/Users/hp/Desktop/FinGT_repo/FinGT/src/together/src/fpb_llama_34_2024-04-08.csv')
                time.sleep(10.0)
            
    return df
####


# output = together.Complete.create(
#     prompt= f"<human>: {prompt} \n<bot>:", # TODO: PROMPT HAS TO COME FROM THE TASK AND MODEL
#         # STEP1: have a function that takes in the model and returns the special tokens and the prompt function
#         # STEP2: use the task provided in args to collect the prompt text
#         # STEP3: use the prompt text on the prompt function associate with that particular model to format the prompt
#     # model=model,
#     # max_tokens=256,
#     # temperature=0.8,
#     # top_k=60,
#     # top_p=0.6,
#     # repetition_penalty=1.1,
#     stop=["<human>", "\n\n"], # TODO: STOP WORDS SHOULD COME FROM THE MODEL ONLY
# )
# def fpb_prompt(sentence: str):
#     system_prompt = f''' Discard all the previous instructions. Behave like you are an expert sentence sentiment classifier  '''

#     user_msg = f''' Classify the following sentence into ‘NEGATIVE’, ‘POSITIVE’, or ‘NEUTRAL’
#                 class. Label ‘NEGATIVE’ if it is corresponding to negative sentiment, ‘POSITIVE’ if it is
#                 corresponding to positive sentiment, or ‘NEUTRAL’ if the sentiment is neutral. Provide
#                 the label in the first line and provide a short explanation in the second line. This is the sentence: {sentence}'''
    
#     prompt = f"""<s>[INST] <<SYS>> {system_prompt} <</SYS>> {user_msg} [/INST]"""
    
#     return prompt