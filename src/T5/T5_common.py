import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import T5Tokenizer

class ECTdataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        source_len: int = 512,
        target_len: int = 128,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.source_len = source_len
        self.target_len = target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        input = row["input"]
        input_encoding = self.tokenizer(
            input,
            max_length=self.source_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        output_encoding = self.tokenizer(
            row["output"],
            max_length=self.target_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        labels = output_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            input=input,
            output=row["output"],
            input_ids=input_encoding["input_ids"].flatten(),
            input_attention_mask=input_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=output_encoding["attention_mask"].flatten(),
        )
