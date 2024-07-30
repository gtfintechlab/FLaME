# main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
import pytorch_lightning as pl
from src.utils.evaluate_ectsum import Evaluate
from src.ECT.ect_dataset import (
    ECTDataModule,
    ECTSumModel,
    summarize,
    iterate_df,
    save_data,
)

# Set seed for reproducibility
pl.seed_everything(42)

# Load data
train_df = pd.read_csv("train.csv")  # Replace with your actual train file
eval_df = pd.read_csv("val.csv")  # Replace with your actual eval file

# Load tokenizer
model_path = "/fintech_3/hf_models/t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Set hyperparameters
N_EPOCHS = 3
BATCH_SIZE = 8

# Create data module
data_module = ECTDataModule(train_df, eval_df, tokenizer, batch_size=BATCH_SIZE)
data_module.setup()

# Create the model
model = ECTSumModel(model_path)

# Train the model
trainer = pl.Trainer(max_epochs=N_EPOCHS)
trainer.fit(model, data_module)

# Load the best checkpoint
trained_model = ECTSumModel.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path
)
trained_model.freeze()

# Evaluate and save results
evaluator = Evaluate()
data = "ectsum_data.csv"
model_name = "T5-base"
results = iterate_df(data, trained_model, tokenizer)
output_path = save_data(data, model_name, results)
evaluator.append_scores(output_path)
