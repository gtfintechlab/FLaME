import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd
import numpy as np
from time import time
from datetime import date


today = date.today()


# load data 
df_data = pd.read_csv('../data/poc_revenue_mcap_data.csv')
df_data = df_data[['CONM', 'year', 'mcap', 'revt']]



# set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str("0")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device assigned: ", device)

model = "meta-llama/Llama-2-7b-chat-hf"
# model = "/hdd/data_8tb_disk/llama_2_models/llama-2-7b-chat"

# get model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model)

tokenizer.add_special_tokens(
    {

        "pad_token": "<PAD>",
    }
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)


for year in range(1980, 2021):

    df_copy = df_data.copy()

    df_copy = df_copy.loc[(df_copy['year'] == year)]
    df_copy = df_copy.reset_index(drop=True)
    # df_copy = df_copy.head(10)
    print(df_copy.shape)
    print(df_copy.head())
    



    prompts_list = []
    for index, row in df_copy.iterrows():
        
        company_name = row['CONM']
        financial_year = year
        message = f'What was the revenue of {company_name} in {financial_year}?'
        prompts_list.append(message)

    
    start_t = time()

    # documentation: https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation
    res = pipeline(
        prompts_list, 
        max_new_tokens=100, 
        do_sample=True, 
        use_cache=True, 
        top_p=1.0, 
        top_k=50, 
        temperature=0.01, 
        num_return_sequences=1, 
        eos_token_id=tokenizer.eos_token_id
        )
    
    output_list = []
    
    for index, row in df_copy.iterrows():
        #print(res[i][0]['generated_text'][len(prompts_list[i]):])
        answer = res[index][0]['generated_text'][len(prompts_list[index]):]
        answer = answer.strip()

        temp_list = list(row)
        temp_list.append(prompts_list[index])
        temp_list.append(answer)

        output_list.append(temp_list)

    results = pd.DataFrame(output_list, columns=['CONM', 'year', 'mcap', 'revt', 'prompt', "prompt_output"])

    time_taken = int((time() - start_t)/60.0)
    results.to_csv(f'../data/llm_prompt_outputs/llama_2_7b_yearly/llama_2_7b_chat_year_{year}_{today.strftime("%d_%m_%Y")}_{time_taken}.csv', index=False)