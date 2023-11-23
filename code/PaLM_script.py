# pip install -U google-generativeai
# pip install pandas
# pip install numpy

from time import sleep, time
from datetime import date
import pandas as pd
import os
import numpy as np
import torch
import google.generativeai as palm
from evaluate_metrics import Evaluate
from LangChain_splitter import split_document


today = date.today()

evaluator = Evaluate()


palm.configure(api_key="AIzaSyAedmPMHxXBrjTQFgOoypvQIYZLiz6L21k")

# Prompt method


def prompter(doc):

    docs = split_document(doc, 1000)
    output_list = []
    for i, doc in zip(range(len(docs)), docs):
        prompt = """Discard all the previous instructions.
        Behave like you are an expert at summarization tasks.
        Below an earnings call transcript of a Russell 3000 Index company
        is provided. Perform extractive summarization followed by
        paraphrasing the transcript in bullet point format according to the
        experts-written short telegram-style bullet point summaries
        derived from corresponding Reuters articles. The target length of
        the summary should be at most 50 words. \n\n"""

        prompt += doc

        res = palm.chat(messages=prompt)
        output_list.append(res.last)

    text = ""
    for t in output_list:
        text = text + "\n\n" + t
    return text


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

    res = palm.chat(messages=prompt)
    return res.last

def iterate_df(data_file):
    df = pd.read_csv(data_file)
    output_list = []
    for i, row in df.iterrows():
        input = row["input"]
        text = generate_text(prompter(input))
        output_list.append(text)

    return output_list


def save_data(data_filename, model_name, generated_output_list):

    df = pd.read_csv(data_filename)

    df["predicted_text"] = generated_output_list

    output_filename = f"{model_name}_{today}_output.csv"
    df.to_csv(output_filename, index=False)
    return output_filename


data_path = "ectsum_data.csv"
model = "PaLM"
results = iterate_df(data_path)
path = save_data(data_path, model, results)
evaluator.append_scores(path)
    
