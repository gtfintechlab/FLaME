import numpy as np
import pandas as pd
import json
import torch

from datasets import load_dataset
from datasets import load_metric

from transformers import AutoConfig
from transformers import AutoModelForCausalLM  # Zero-shot LLaMA-2-7B
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline
from transformers import TrainingArguments
from transformers import Trainer

from evaluate_metrics import Evaluate
from Document_splitter import split_document


model = "/fintech_3/hf_models/Llama-2-7b-chat-hf"


tokenizer = AutoTokenizer.from_pretrained(model)

# Set pipeline for text generation
pipeline_obj = pipeline(
    "text-generation",
    model=model,
    # tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)


def llama_prompter(doc):
    docs = split_document(doc, 1000)

    # Create prompt
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

        # Chat with model through prompt
        res = pipeline_obj(
            prompt,
            max_new_tokens=64,
            do_sample=True,
            num_return_sequences=1,
            # eos_token_id=tokenizer.eos_token_id,
        )
        output_list.append(str(res))

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

    res = pipeline_obj(
        prompt,
        max_new_tokens=64,
        do_sample=True,
        num_return_sequences=1,
        # eos_token_id=tokenizer.eos_token_id,
    )
    return str(res)


def iterate_df(data_file):
    df = pd.read_csv(data_file)
    output_list = []
    for i, row in df.iterrows():
        input = row["input"]

        text = generate_text(llama_prompter(input))

        # output_text = formatter(row['output'],text)
        print(text)

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
model = "Llama-2-7b-chat-hf"
results = iterate_df(data)
path = save_data(data, model, results)
evaluator.append_scores(path)
