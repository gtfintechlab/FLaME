# pip install openai
import json
import openai
import pandas as pd
from Document_splitter import document_splitter
from evaluate_metrics import Evaluate


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
        presence_penalty=0,
    )["choices"][0]["text"]
    return resp


def generate_df(doc):

    output_list = []
    prompt = """Discard all the previous instructions.
    Behave like you are an expert at summarization tasks.
    Below an earnings call transcript of a Russell 3000 Index company
    is provided. Perform extractive summarization followed by
    paraphrasing the transcript in bullet point format according to the
    experts-written short telegram-style bullet point summaries
    derived from corresponding Reuters articles. The target length of
    the summary should be at most 50 words. \n\n"""

    prompt += doc
    response = chat_gpt(prompt)
    output_list.append(response)

    return output_list


def generate_text(input_text):

    prompt = """Discard all the previous instructions.
    Behave like you are an expert at summarization tasks.
    Given below is a combination of different summaries from the same Earnings Call Transcript.
    Perform extractive summarization followed by
    paraphrasing the summaries as one in bullet point format according to the
    experts-written short telegram-style bullet point summaries
    derived from corresponding Reuters articles. The target length of
    the summary should be at most 50 words \n\n"""

    prompt += input_text

    res = chat_gpt(prompt)
    return res


def iterate_df(data_file):
    df = pd.read_csv(data_file)
    output_list = []
    for i, row in df.iterrows():
        input = row["input"]
        text = generate_df((input))
        output_list.append(text)

    return output_list


def save_data(data_filename, model_name, generated_output_list):

    df = pd.read_csv(data_filename)

    df["predicted_text"] = generated_output_list

    output_filename = f"{model_name}_output.csv"
    df.to_csv(output_filename, index=False)
    return output_filename


evaluator = Evaluate()
data = "ectsum_data.csv"
model = "GPT-4"
results = iterate_df(data)
path = save_data(data, model, results)
evaluator.append_scores(path)
