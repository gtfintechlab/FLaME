#pip install openai
import json
import openai
import pandas as pd
from LangChain_splitter import document_splitter
import sys
sys.path.append("/Users/hp/Desktop/ZeroNotHero/ZeroNotHero")
import evaluate_metrics


openai.api_key = "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"
decoder = json.JSONDecoder()

def chat_gpt(prompt_text):
    resp = openai.Completion.create(
                model="gpt-3.5-turbo",
                prompt=prompt_text,
                temperature=0,
                max_tokens=8000,
                top_p=1.0,
                frequency_penalty=0,
                presence_penalty=0
            )['choices'][0]['text']
    return resp

def generate_df(doc):
    
    output_list = []
    prompt = '''Discard all the previous instructions.
    Behave like you are an expert at summarization tasks.
    Below an earnings call transcript of a Russell 3000 Index company
    is provided. Perform extractive summarization followed by
    paraphrasing the transcript in bullet point format according to the
    experts-written short telegram-style bullet point summaries
    derived from corresponding Reuters articles. The target length of
    the summary should be at most 50 words. \n\n'''
        
    prompt += doc
    response = chat_gpt(prompt)
    output_list.append(response)
    
    return output_list


#def generate_text(input_text):
    
    prompt = '''Discard all the previous instructions.
    Behave like you are an expert at summarization tasks.
    Given below is a combination of different summaries from the same Earnings Call Transcript.
    Perform extractive summarization followed by
    paraphrasing the summaries as one in bullet point format according to the
    experts-written short telegram-style bullet point summaries
    derived from corresponding Reuters articles. The target length of
    the summary should be at most 50 words \n\n'''

    prompt += input_text

    res = palm.chat(messages = prompt)
    return res.last
    
    

#def formatter(reference_text, predicted_text):
    
    prompt = '''Discard all the previous instructions.
                Given the following text format:\n\n'''
    
    prompt += reference_text

    prompt += '''\n\nFormat the following text in the same format with the same number of sentences.
                    Make it concise if needed retaining the main financial takeaways.\n\n'''

    prompt += predicted_text


    res = palm.chat(messages = prompt)
    return(res.last)   
    
def iterate_df(data_file) :
    df = pd.read_csv(data_file)
    output_list = []
    for i,row in df.iterrows():
        input = row["input"]
        
        text = generate_df((input))
        
        #output_text = formatter(row['output'],text)
        print(text)
        
        output_list.append(text)
        
    return output_list

    
def save_data(data_filename, model_name, generated_output_list):
    
    df = pd.read_csv(data_filename)

    df['predicted_text'] = generated_output_list

    output_filename = f"{model_name}_output.csv"
    df.to_csv(output_filename, index=False)
    return output_filename
    #print(f"Processed data saved to {output_filename}")


    #data = "ectsum_data.csv"
data = "/Users/hp/Desktop/ZeroNotHero/sample_tester.csv"
model = "GPT-4"
results = iterate_df(data)
path = save_data(data,model,results)
evaluate_metrics.append_scores(path)


