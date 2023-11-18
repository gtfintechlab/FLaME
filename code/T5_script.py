#pip install transformers
#pip install datasets
#pip install accelerate

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split

#from termcolor import colored

import textwrap

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer)
from tqdm import tqdm

pl.seed_everything(42)

df = pd.read_csv("/fintech_3/Huzaifa/train.csv", nrows=50)


df = df.dropna()
train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)

class ECTdataset(Dataset):
    def __init__(self,
                data: pd.DataFrame,
                tokenizer: T5Tokenizer,
                source_len: int = 512,
                target_len: int = 128):
        self.tokenizer = tokenizer
        self.data = data
        self.source_len = source_len
        self.target_len = target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        input = row["input"]
        input_encoding = tokenizer(
            input,
            max_length=self.source_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        output_encoding = tokenizer(
            row["output"],
            max_length=self.target_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        labels = output_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            input = input,
            output = row["output"],
            input_ids = input_encoding["input_ids"].flatten(),
            input_attention_mask = input_encoding["attention_mask"].flatten(),
            labels = labels.flatten(),
            labels_attention_mask = output_encoding["attention_mask"].flatten()
        )

class ECTDataModule(pl.LightningDataModule):
    def __init__(self,
                train_df: pd.DataFrame,
                eval_df: pd.DataFrame,
                tokenizer: T5Tokenizer,
                batch_size: int = 8,
                source_len: int = 512,
                target_len: int = 128):
        super().__init__()
        self.train_df = train_df
        self.eval_df = eval_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.source_len = source_len
        self.target_len = target_len

    def setup(self, stage=None):
        self.train_dataset = ECTdataset(self.train_df, self.tokenizer, self.source_len, self.target_len)
        self.test_dataset = ECTdataset(self.eval_df, self.tokenizer, self.source_len, self.target_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,shuffle = False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,shuffle = False, num_workers=2)

Model = "/fintech_3/hf_models/t5-base"
tokenizer = T5Tokenizer.from_pretrained(Model)

N_EPOCHS = 3
BATCH_SIZE = 8

data_module = ECTDataModule(train_df, eval_df, tokenizer, batch_size=BATCH_SIZE)
data_module.setup()

class ECTSumModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(Model,return_dict = True)

    def forward(self, input_ids, attention_mask,decoder_attanetion_mask, labels=None):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attanetion_mask
        )
        return outputs.loss, outputs.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["input_attention_mask"]
        labels = batch["labels"]
        decoder_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(input_ids, attention_mask, decoder_attention_mask, labels)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["input_attention_mask"]
        labels = batch["labels"]
        decoder_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(input_ids, attention_mask, decoder_attention_mask, labels)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss


    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["input_attention_mask"]
        labels = batch["labels"]
        decoder_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(input_ids, attention_mask, decoder_attention_mask, labels)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.0001)
        return optimizer

model = ECTSumModel()

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

logger = TensorBoardLogger("lightning_logs", name="ect",default_hp_metric=False)

trainer = pl.Trainer(
    max_epochs=N_EPOCHS,
    callbacks=checkpoint_callback,
    logger=logger # Use this argument to specify the number of GPUs
)

trainer.fit(model, data_module)

trained_model = ECTSumModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
trained_model.freeze()

def summarize(text):
    text_encoding = tokenizer(text,
                max_length = 512,
                padding = "max_length",
                truncation = True,
                return_attention_mask = True,
                add_special_tokens = True,
                return_tensors = "pt")
    text_encoding.to("cuda:0")

    generated_ids = trained_model.model.generate(
        input_ids = text_encoding["input_ids"],
        attention_mask = text_encoding["attention_mask"],
        max_length = 150,
        min_length = 50,
        num_beams = 2,
        early_stopping = True
    )

    preds = [
        tokenizer.decode(gen_ids,skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for gen_ids in generated_ids
    ]

    return "".join(preds)

dftrain = pd.read_csv("/fintech_3/Huzaifa/train.csv")
dftest = pd.read_csv("/fintech_3/Huzaifa/test.csv")

dfoutput = pd.DataFrame(columns = ['input','output'])

'''for index, row in dftrain.iterrows():
    input_value = row['input']
    output_value = summarize(input_value)
    
    dfoutput = dfoutput.append({'input': input_value, 'output': output_value}, ignore_index=True)
    
    print(output_value)
    
for index, row in dftest.iterrows():
    input_value = row['input']
    output_value = summarize(input_value)
    
    dfoutput = dfoutput.append({'input': input_value, 'output': output_value}, ignore_index=True)
    
    print(output_value)

output_path = "t5_output.csv"

dfoutput.to_csv(output_path,index = False)'''
inp = dftrain[0]['input']
out = summarize(inp)
print(out)