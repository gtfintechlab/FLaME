#!/usr/bin/env python
# coding: utf-8
# ***!!!TODO!!!***
# change to using https://huggingface.co/datasets/gtfintechlab/fomc-example-dataset
# - remove wandb logging from conference notebook, switch to tensorboard
# - Determine if its possible to just checkpoint and log adapters to wandb
# - MAKE CONFERENCE NOTEBOOK GENERIC HF WITH NO TOKEN, USE PUBLIC DATASET


# ====================== IMPORTS ======================
# Standard Libraries
import os
import sys
import uuid
import logging
import warnings
import pprint
from pathlib import Path
from collections import namedtuple
from datetime import datetime
from functools import partial

# Third-Party Libraries
import pandas as pd
import torch
import torch.nn as nn
import wandb
import huggingface_hub
import bitsandbytes as bnb
from IPython.display import display
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset

# Transformers and Custom Libraries
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline,
    Trainer, TrainingArguments, logging as hf_logging
)
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM, PeftModel

# ====================== HUGGINGFACE ======================
# TODO: Remove HF TOKEN from final version
HF_AUTH = "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"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ====================== WEIGHTS AND BIASES ======================
# TODO: Remove WANDB TOKEN from final version

import wandb

WANDB_PROJECT = f"llama2_sft_fomc"
# Set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
# Turn off save your trained model checkpoint to wandb (our models are too large)
os.environ["WANDB_LOG_MODEL"] = "false"
# Turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"
os.environ["WANDB_API_KEY"] = "fa69ffc6a97578da0410b553042cbb8b3bf5fcaf"
os.environ["WANDB_NOTEBOOK_NAME"] = f"llama2_sft"
wandb.login()

# # Convert Args namedtuple to dictionary
# args_dict = args._asdict()

# In[23]:


# ====================== USER PARAMETERS ======================
organization = "gtfintechlab"
report_to="tensorboard"
logging_dir="/home/AD/gmatlin3/tensorboard/logs"

# ====================== TASK PARAMETERS ======================
task_name = "fomc_communication"
seeds = (5768, 78516, 944601)
seed = seeds[0]

# ====================== MODEL PARAMETERS ======================
model_parameters = "7b"
model_id = f"meta-llama/Llama-2-{model_parameters}-chat-hf"
model_name = model_id.split("/")[-1]

# ====================== PROMPT PARAMETERS ======================
system_prompt = f"""Discard all the previous instructions.
Below is an instruction that describes a task.
Write a response that appropriately completes the request.
"""

instruction_prompt = f"""Behave like you are an expert sentence classifier.
Classify the following sentence from FOMC into 'HAWKISH', 'DOVISH', or 'NEUTRAL' class.
Label 'HAWKISH' if it is corresponding to tightening of the monetary policy.
Label 'DOVISH' if it is corresponding to easing of the monetary policy.
Label 'NEUTRAL' if the stance is neutral.
Provide a single label from the choices 'HAWKISH', 'DOVISH', or 'NEUTRAL' then stop generating text.

The sentence: "
"""

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"

repo_name = f"{organization}/{model_name}_{task_name}"

# ====================== QLORA PARAMETERS ======================
# LoRA attention dimension
lora_r = 64
# Alpha parameter for LoRA scaling
lora_alpha = 16
# Dropout probability for LoRA layers
lora_dropout = 0.1

# ====================== SFT PARAMETERS ======================
# Maximum sequence length to use
max_seq_length = None
# Pack multiple short examples in the same input sequence to increase efficiency
packing = False
neftune_noise_alpha = 5

# ====================== CUDA PARAMETERS ======================
# Enable fp16/bf16 training
compute_dtype = torch.bfloat16
fp16, bf16 = False, True

CUDA_N_GPUS = torch.cuda.device_count()
CUDA_MAX_MEMORY = f"{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB"
CUDA_MAX_MEMORY = {i: CUDA_MAX_MEMORY for i in range(CUDA_N_GPUS)}
logger.info(f"Using k={CUDA_N_GPUS} CUDA GPUs with max memory {CUDA_MAX_MEMORY}")

# device_map = {"": 0} # Load the entire model on the GPU 0
device_map = "auto" # Automatically determine the device map

save_safetensors = True

# ====================== BITSANDBYTES PARAMETERS ======================
# Activate 4-bit precision base model loading
load_in_4bit = True

# Activate 8-bit precision base model loading
load_in_8bit = False

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = compute_dtype

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
bnb_4bit_use_double_quant = False

# ====================== TRAININGARGUMENTS PARAMETERS ======================
# Output directory where the model predictions and checkpoints will be stored
output_dir = Path(f"/fintech_3/20231018/results/{model_name}_{task_name}")

# Number of training epochs
num_train_epochs = 1

# Batch size per GPU for training
per_device_train_batch_size = 8

# Batch size per GPU for evaluation
per_device_eval_batch_size = 8

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = False

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "adamw_bnb_8bit"

# Learning rate schedule
lr_scheduler_type = "constant"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 200

# Log every X updates steps
logging_steps = 25

load_best_model_at_end = True

strategy="steps"
save_strategy=strategy
logging_strategy=strategy
evaluation_strategy=strategy

disable_tqdm=True
predict_with_generate=True



# ====================== LOGGING SETUP ======================
def setup_logging():
    logger = logging.getLogger("llama2_finetune")
    logger.setLevel(logging.DEBUG)
    hf_logging.set_verbosity(hf_logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler("llama2_finetune.log")
    c_handler.setLevel(logging.DEBUG)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    format = "%(name)s - %(levelname)s - %(message)s"
    c_format = logging.Formatter(format)
    f_format = logging.Formatter(format)
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger

logger = setup_logging()

# ====================== ARGUMENTS SETUP ======================
def setup_args():
    Args = namedtuple(
        "Args",
        # ... (all your arguments here)
    )

    args = Args(
        task_name=task_name,
        system_prompt=system_prompt,
        # ... (all your argument values here)
    )

    return args

args = setup_args()
