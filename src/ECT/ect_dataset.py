import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import T5Tokenizer

class ECTdataset(Dataset):
    # Existing code for ECTdataset class...
    
class ECTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = 8,
        source_len: int = 512,
        target_len: int = 128,
    ):
        super().__init__()
        self.train_df = train_df
        self.eval_df = eval_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.source_len = source_len
        self.target_len = target_len

    def setup(self, stage=None):
        self.train_dataset = ECTdataset(
            self.train_df, self.tokenizer, self.source_len, self.target_len
        )
        self.test_dataset = ECTdataset(
            self.eval_df, self.tokenizer, self.source_len, self.target_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

class ECTSumModel(pl.LightningModule):
    def __init__(self, model_path):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path, return_dict=True
        )
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path, return_dict=True
        )

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
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
        input_ids = batch["input_ids"]
        attention_mask = batch["input_attention_mask"]
        labels = batch["labels"]
        decoder_attention_mask = batch["labels_attention_mask"]
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
