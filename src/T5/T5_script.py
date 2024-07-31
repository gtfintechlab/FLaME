# pip install transformers
# pip install datasets
# pip install accelerate
# pip install torch
# pip install pytorch-lightning

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from src.ECT.ect_dataset import ECTdataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers import T5ForConditionalGeneration, T5Config

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from utils.evaluate_ectsum import EvaluateMetrics


from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
)
from tqdm import tqdm

pl.seed_everything(42)

from src.ECT.ect_dataset import load_data, ECTDataModule

train_df, eval_df = load_data("ectsum_data.csv")







Model = "/fintech_3/hf_models/t5-base"
tokenizer = T5Tokenizer.from_pretrained(Model)

N_EPOCHS = 3
BATCH_SIZE = 8

data_module = ECTDataModule(train_df, eval_df, tokenizer, batch_size=BATCH_SIZE)
data_module.setup()




from src.ECT.ect_dataset import ECTSumModel

model = ECTSumModel(Model)

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min",
)

logger = TensorBoardLogger("lightning_logs", name="ect", default_hp_metric=False)
# logger = CSVLogger("logs", name="ect")

trainer = pl.Trainer(
    max_epochs=N_EPOCHS,
    callbacks=checkpoint_callback,
    logger=logger,  # Use this argument to specify the number of GPUs
)

trainer.fit(model, data_module)

trained_model = ECTSumModel.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path
)
trained_model.freeze()


def summarize(text):
    text_encoding = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_encoding.to("cuda:0")

    generated_ids = trained_model.model.generate(
        input_ids=text_encoding["input_ids"],
        attention_mask=text_encoding["attention_mask"],
        max_length=150,
        min_length=50,
        num_beams=2,
        early_stopping=True,
    )

    preds = [
        tokenizer.decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        for gen_ids in generated_ids
    ]

    return "".join(preds)


def iterate_df(data_file):
    df = pd.read_csv(data_file)
    output_list = []
    for i, row in df.iterrows():
        input = row["input"]

        text = summarize(input)

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
model = "T5-base"
results = iterate_df(data)
path = save_data(data, model, results)
evaluator.append_scores(path)
