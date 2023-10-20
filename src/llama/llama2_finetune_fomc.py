#!/usr/bin/env python
# coding: utf-8

# change to using https://huggingface.co/datasets/gtfintechlab/fomc-example-dataset

# # Not for Public
# 
# ***!!!TODO!!!***
# - remove wandb logging from conference notebook, switch to tensorboard
# - Determine if its possible to just checkpoint and log adapters to wandb
# - MAKE CONFERENCE NOTEBOOK GENERIC HF WITH NO TOKEN, USE PUBLIC DATASET

# In[1]:


import os
import huggingface_hub

HF_AUTH = "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"
huggingface_hub.login(token=HF_AUTH)

import wandb

WANDB_PROJECT = f"llama2_sft_fomc"

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"] = WANDB_PROJECT

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "false" # dont log any models
# os.environ["WANDB_LOG_MODEL"] = "checkpoint" # log all model checkpoints


# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"
os.environ["WANDB_API_KEY"] = "fa69ffc6a97578da0410b553042cbb8b3bf5fcaf"
os.environ["WANDB_NOTEBOOK_NAME"] = f"llama2_sft"

wandb.login()

# # Convert Args namedtuple to dictionary
# args_dict = args._asdict()

# run = wandb.init(
#     project=WANDB_PROJECT,
#     config=args_dict  # Passing the converted dictionary as config
# )

import uuid

def generate_uid(id_length=8, dt_format="%y%m%d"):
    date_str = datetime.now().strftime(dt_format)

    # Generate a short UUID
    uid = str(uuid.uuid4())[:id_length]

    # Combine
    uid = f"{uid}_{date_str}"

    return uid
# # Supervised Fine-Tuning of Llama2 on FOMC

# ## Setup

# ### Imports

# In[18]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[19]:


import os
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display

from tqdm.notebook import tqdm
from transformers import GenerationConfig

SRC_DIRECTORY = Path().cwd().resolve().parent

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


# In[20]:


import gc
import logging
from transformers import logging as hf_logging

hf_logging.set_verbosity(hf_logging.DEBUG)

logger = logging.getLogger("llama2_finetune")
logger.setLevel(logging.DEBUG)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('llama2_finetune.log')
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)


# Create formatters and add it to handlers
format = '%(name)s - %(levelname)s - %(message)s'
c_format = logging.Formatter(format)
f_format = logging.Formatter(format)
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)


# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)


# In[21]:


import pprint

pp = pprint.PrettyPrinter(indent=4)

import warnings
from collections import namedtuple
from datetime import datetime
from functools import partial

import bitsandbytes as bnb
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, load_dataset
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    LlamaTokenizer,
    TextGenerationPipeline,
    Trainer,
    TrainingArguments,
    logging,
    pipeline,
)

# Related to HuggingFace's Tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from trl import SFTTrainer

from peft import PeftModel


# In[22]:


from transformers.trainer_callback import TrainerCallback

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


# ### Configuration

# In[23]:


################################################################################
# User parameters
################################################################################
organization = "gtfintechlab"
report_to="tensorboard"
logging_dir="/home/AD/gmatlin3/tensorboard/logs"

################################################################################
# Task parameters
################################################################################
task_name = "fomc_communication"
seeds = (5768, 78516, 944601)
seed = seeds[0]

################################################################################
# Model parameters
################################################################################
model_parameters = "7b"
model_id = f"meta-llama/Llama-2-{model_parameters}-chat-hf"
model_name = model_id.split("/")[-1]

################################################################################
# Prompt parameters
################################################################################
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

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

neftune_noise_alpha = 5

################################################################################
# CUDA Parameters
################################################################################

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

################################################################################
# bitsandbytes parameters
################################################################################

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

################################################################################
# TrainingArguments parameters
################################################################################

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


# In[24]:


Args = namedtuple(
    "Args",
    [
        "task_name",
        "system_prompt",
        "instruction_prompt",
        "seed",
        "model_id",
        "model_name",
        "organization",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "max_seq_length",
        "packing",
        "device_map",
        "load_in_4bit",
        "load_in_8bit",
        "bnb_4bit_compute_dtype",
        "bnb_4bit_use_double_quant",
        "bnb_4bit_quant_type",
        "output_dir",
        "num_train_epochs",
        "fp16",
        "bf16",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "max_grad_norm",
        "learning_rate",
        "weight_decay",
        "optim",
        "lr_scheduler_type",
        "max_steps",
        "warmup_ratio",
        "group_by_length",
        "save_steps",
        "save_strategy",
        "logging_strategy",
        "logging_steps",
        "evaluation_strategy",
        "neftune_noise_alpha",
        "save_safetensors",
        "load_best_model_at_end",
        "disable_tqdm",
        "B_INST",
        "E_INST",
        "B_SYS",
        "E_SYS",
        "BOS",
        "EOS",
        "report_to",
        "logging_dir",
        "predict_with_generate",
    ],
)

args = Args(
    task_name=task_name,
    system_prompt = system_prompt,
    instruction_prompt = instruction_prompt,
    seed=seed,
    model_id=model_id,
    model_name=model_id.split("/")[-1],
    organization=organization,
    lora_r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    max_seq_length=max_seq_length,
    packing=packing,
    device_map=device_map,
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    fp16=fp16,
    bf16=bf16,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    max_grad_norm=max_grad_norm,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    optim=optim,
    lr_scheduler_type = lr_scheduler_type,
    max_steps=max_steps,
    warmup_ratio= warmup_ratio,
    group_by_length=group_by_length,
    save_steps=save_steps,
    save_strategy=save_strategy,
    evaluation_strategy=evaluation_strategy,
    logging_strategy=logging_strategy,
    logging_steps=logging_steps,
    neftune_noise_alpha=neftune_noise_alpha,
    save_safetensors=save_safetensors,
    load_best_model_at_end=load_best_model_at_end,
    disable_tqdm=disable_tqdm,
    B_INST = B_INST,
    E_INST = E_INST,
    B_SYS = B_SYS,
    E_SYS = E_SYS,
    BOS = BOS,
    EOS = EOS,
    report_to=report_to,
    logging_dir=logging_dir,
    predict_with_generate=predict_with_generate,
)


# ### Functions

# In[25]:


def log_trainable_parameters(model, logger):
    """
    Logs the number of trainable parameters in the model.
    
    Parameters:
    model : torch.nn.Module
        The model whose parameters we want to log.
    logger : logging.Logger
        Logger object to log the information.
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    # Logging the information instead of printing
    logger.info(
        f"Trainable params: {trainable_params} || All params: {all_params} || Trainable%: {100 * trainable_params / all_params}"
    )


def log_dtypes(model, logger):
    """
    Logs the data types of the model parameters and their proportions.
    
    Parameters:
    model : torch.nn.Module
        The model whose parameter data types we want to log.
    logger : logging.Logger
        Logger object to log the information.
    """
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    
    total = sum(dtypes.values())
    
    # Logging the information instead of printing
    for dtype, count in dtypes.items():
        logger.info(f"{dtype}: {count} ({100 * count / total:.2f}%)")


def create_prompt_format(
    sample,
    args: Args,
    context_field="sentence",
    response_field="label_decoded"
):
    if instruction_prompt is None or not isinstance(instruction_prompt, str) or instruction_prompt.strip() == "":
        raise ValueError(f"Invalid instruction prompt received: {instruction_prompt}. It must be a non-empty string.")
    if system_prompt is None or not isinstance(system_prompt, str) or system_prompt.strip() == "":
        raise ValueError(f"Invalid system prompt received: {system_prompt}. It must be a non-empty string.")

    if not sample or not all(
        isinstance(sample[field], str) for field in [context_field, response_field]
    ):
        raise ValueError("Fields must be a non-empty strings.")

    prompt = (
        args.B_INST
        + args.B_SYS
        + args.system_prompt
        + args.E_SYS
        + args.instruction_prompt
        + sample[context_field]
        + args.E_INST
    )
    sample["text"] = str(prompt[0])
    return sample


def preprocess_batch(batch, tokenizer, max_seq_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_seq_length,
        truncation=True,
    )


def preprocess_dataset(
    args: Args,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    dataset: Dataset
):

    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    seed = args.seed
    
    # Add prompt to each sample
    print("Preprocessing dataset...")

    _prompt_format_function = partial(
        create_prompt_format, args=args
    )
    
    dataset = dataset.map(
        _prompt_format_function,
        batched=False
    )
    
    
    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(
        preprocess_batch, max_seq_length=max_seq_length, tokenizer=tokenizer
    )

    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_seq_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

def find_all_linear_names(model):
    # SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    cls = (
        bnb.nn.Linear4bit
    )  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

def create_peft_config(args:Args, modules):
    peft_config = LoraConfig(
        # Pass our list as an argument to the PEFT config for your model
        target_modules=modules,
        # Dimension of the LoRA matrices we update in adapaters
        r=args.lora_r,
        # Alpha parameter for LoRA scaling
        lora_alpha=args.lora_alpha,
        # Dropout probability for LoRA layers
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return peft_config


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

def merge_evaluation_results(baseline_results, final_results):
    all_metrics = set(baseline_results.keys()).union(final_results.keys())

    data = {
        'Metric': list(all_metrics),
        'Baseline': [baseline_results.get(metric, None) for metric in all_metrics],
        'After Fine-tuning': [final_results.get(metric, None) for metric in all_metrics]
    }

    return pd.DataFrame(data)


# ## Supervised Fine-Tuning
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}
# In[26]:


import evaluate


# In[27]:


def metric_computer(tokenizer):
    bleu_metric = evaluate.load('bleu')
    rouge_metric = evaluate.load('rouge')

    def compute_metrics(p):
        """
        This function receives a datasets.arrow_dataset.Dataset instance that contains the model's predictions and labels
        and computes the custom metrics (BLEU, ROUGE).
        """
        predictions, references = p
        
        # Decode the logits to get predicted token ids
        pred_ids = np.argmax(p.predictions, axis=2)

        # Decode the predicted and label ids token ids to text
        pred_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids]
        label_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in p.label_ids]

        # You might need to preprocess predictions and labels, depending on your needs
        # For example, you might want to decode the predicted token ids to text
        # print("p:",p)
        # print("Predictions:", p.predictions)
        # print("Labels:", p.label_ids)

        # Calculate Metrics
        bleu_score = bleu_metric.compute(predictions=pred_texts, references=label_texts)
        rouge_score = rouge_metric.compute(predictions=pred_texts, references=label_texts)

        # Combine the metrics in a dictionary and return
        return {
            'bleu': bleu_score,
            'rouge': rouge_score,
        }
    return compute_metrics


# In[28]:


def train(args):
    logger.info("Starting the training process...")
    logger.info("Creating BitsAndBytesConfig...")
    bnb_config = BitsAndBytesConfig(
        # Activate k-bit precision base model loading
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,

        # Activate nested quantization for 4-bit base models (double quantization)
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_8bit_use_double_quant=args.bnb_4bit_use_double_quant,
        
        # Quantization type (fp4 or nf4)
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_8bit_quant_type=args.bnb_4bit_quant_type,
        
        # Compute dtype for 4-bit base models
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_8bit_compute_dtype=args.bnb_4bit_compute_dtype,
        
    )

    logger.info("Loading the Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=False)
    logger.info("Tokenizer pad token specified as the EOS token")
    tokenizer.pad_token = EOS
    # logger.info("Tokenizer configured to fix overflow issues with fp16 training"
    # tokenizer.padding_side = "right"
    
    compute_metrics_function = metric_computer(tokenizer)
    logger.info(compute_metrics_function)

    logger.info("Loading the CausalLM...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        device_map=args.device_map,
        max_memory=CUDA_MAX_MEMORY,
        torch_dtype=compute_dtype,
        quantization_config=bnb_config,
        trust_remote_code=False
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    max_seq_length = get_max_length(model)
    logger.info(f"Model Dtypes after loading ...")
    log_dtypes(model, logger)

    logger.info("Loading train dataset...")
    train_dataset = load_dataset(
        f"{args.organization}/{args.task_name}", str(args.seed)
    )["train"]

    logger.info("Preprocessing train dataset...")
    preprocessed_train_dataset = preprocess_dataset(
        args=args,
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    logger.info("Train dataset preprocessed.")

    logger.info("Loading test dataset...")
    test_dataset = load_dataset(f"{args.organization}/{args.task_name}", str(args.seed))[
        "test"
    ]
    logger.info("Preprocessing test dataset...")
    preprocessed_test_dataset = preprocess_dataset(
        args=args,
        dataset=test_dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    logger.info("Test dataset preprocessed.")

    logger.info("Getting the model's memory footprint...")
    logger.info(model.get_memory_footprint())
    logger.info("Using the prepare_model_for_kbit_training method from PEFT...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    logger.info(f"Model Dtypes after preparing for kbit training ...")
    log_dtypes(model, logger)
    logger.info("Get lora module names...")
    layers_for_adapters = find_all_linear_names(model)
    logger.info(f"Layers for PEFT Adaptation: {layers_for_adapters}")
    logger.info("Create PEFT config for these modules and wrap the model to PEFT...")
    peft_config = create_peft_config(args, layers_for_adapters)
    logger.info(f"Model Dtypes before PEFT model ...")
    log_dtypes(model, logger)
    model = get_peft_model(model, peft_config)
    logger.info(f"Model Dtypes after PEFT model ...")
    log_dtypes(model, logger)
    logger.info("Information about the percentage of trainable parameters...")
    log_trainable_parameters(model, logger)

    logger.info("Make output directory...")
    output_dir = args.output_dir / "final_checkpoint"
    output_dir.mkdir(mode=0o777, parents=True, exist_ok=True)

    logger.info("Define TrainingArguments...")
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        fp16=args.fp16,
        bf16=args.bf16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm = args.max_grad_norm,
        weight_decay = args.weight_decay,
        optim=args.optim,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        save_safetensors=args.save_safetensors,
        load_best_model_at_end=args.load_best_model_at_end,
        push_to_hub=False,
        evaluation_strategy=args.evaluation_strategy,
        logging_dir=logging_dir,
        report_to=args.report_to,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        group_by_length = args.group_by_length
    )

    logger.info("Defining SFTTrainer...")
    callbacks = [PeftSavingCallback()]
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        packing=args.packing,
        max_seq_length=max_seq_length,
        train_dataset=preprocessed_train_dataset,
        eval_dataset=preprocessed_test_dataset,
        peft_config=peft_config,
        callbacks=callbacks,
        dataset_text_field="text",
        neftune_noise_alpha=args.neftune_noise_alpha,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.predict_with_generate = args.predict_with_generate

    # Evaluate the model before fine-tuning to get the baseline performance
    logger.info("Evaluating the baseline performance of the model before fine-tuning...")
    baseline_results = trainer.evaluate()
    logger.info(f"Baseline evaluation results: {baseline_results}")
  
    logger.info("Running trainer.train() ...")
    trainer.train()
    if args.report_to == 'wandb':
        wandb.finish()

    try:
        metrics = trainer.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
    except Exception as e:
        logger.debug("metrics block failed")
        logger.error(e)

    logger.info("trainer.evaluate() ...")
    final_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {final_results}")

    results_df = merge_evaluation_results(baseline_results, final_results)
    display(results_df)
    
    
    logger.info("trainer.save_state() ...")
    trainer.save_state()
    
    logger.info("Saving tokenizer and last checkpoint of the model...")
    tokenizer.save_pretrained(output_dir)

    model = trainer.model
    log_dtypes(model, logger)
    model.save_pretrained(output_dir)


# ## Run Train

# In[29]:


try:
    train(args)
except Exception as e:
    print(e)
    # Empty VRAM
    if 'trainer' in locals() or 'trainer' in globals():
        del trainer
    if 'model' in locals() or 'model' in globals():
        del model
    if 'pipe' in locals() or 'pipe' in globals():
        del pipe
    torch.cuda.empty_cache()
    gc.collect()
    gc.collect()


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir logs')


# ---

# ---

# ---

# ### Save Results

# In[ ]:


# reload final model checkpoint and save
new_model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir / "final_checkpoint",
    device_map=args.device_map,
    torch_dtype=compute_dtype,
)


# In[ ]:


log_dtypes(new_model, logger)


# In[ ]:


# load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=args.device_map,
    max_memory=CUDA_MAX_MEMORY,
    torch_dtype=compute_dtype,
)


# In[ ]:


log_dtypes(base_model, logger)


# In[ ]:


# This method merges the LoRa layers into the base model. This is needed to use it as a standalone model.
peft_model = PeftModel.from_pretrained(base_model, args.output_dir / "final_checkpoint")
peft_model = peft_model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(
    args.output_dir / "final_checkpoint", pad_token=EOS
)


# In[ ]:


log_dtypes(peft_model, logger)


# In[ ]:


# save inference
merged_checkpoint_dir = args.output_dir / "final_merged_checkpoint"
peft_model.save_pretrained(merged_checkpoint_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_checkpoint_dir)


# ---

# In[ ]:


# push to hub
peft_model.push_to_hub(repo_name, private=True, use_temp_dir=True)
tokenizer.push_to_hub(repo_name, private=True, use_temp_dir=True)


# ---

# ---

# ---
## Evaluation# TODO: move to configs or args
temperature = 0.0  # [0.0, 1.0]; 0.0 means greedy sampling
do_sample = False
max_new_tokens = 256
top_k = 10
top_p = 0.92
repetition_penalty = 1.0  # 1.0 means no penalty
num_return_sequences = 1  # Only generate one response
num_beams = 1


def generate(model=None, tokenizer=None, dataset=None):
    input_ids = tokenizer(dataset["text"])

    # Ensure that input_ids is a PyTorch tensor
    # input_ids = torch.tensor(input_ids).long()

    # Move the tensor to the GPU
    input_ids = input_ids.cuda()

    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(
            temperature=temperature,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            return_dict_in_generate=True,
            output_scores=False,
        ),
    )
    seq = generation_output.sequences
    output = tokenizer.decode(seq[0])
    return output.split("[/INST]")[-1].strip()test_dataset = load_dataset(f"{args.organization}/{args.task_name}", str(args.seed))[
    "test"
]### Baselinebase_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=args.device_map,
    max_memory=CUDA_MAX_MEMORY,
    torch_dtype=compute_dtype,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = EOS
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

max_length = get_max_length(model)

preprocessed_test_dataset = preprocess_dataset(
    tokenizer=tokenizer, max_length=max_length, seed=args.seed, dataset=test_dataset
)# Run text generation pipeline with our next model
prompt = "What is a large language model?"
pipe = pipeline(
    task="text-generation", model=model, tokenizer=tokenizer, max_length=max_length
)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]["generated_text"])# N_GENS = preprocessed_test_dataset.num_rows
N_GENS = 10

output_list = []
for i in range(N_GENS):
    output_list.append(
        generate(model=model, tokenizer=tokenizer, dataset=preprocessed_test_dataset)
    )output_list

.replace(
            "</s>", ""### Supervised Fine-Tuningmodel = AutoModelForCausalLM.from_pretrained(
    merged_checkpoint_dir,
    device_map=args.device_map,
    max_memory=CUDA_MAX_MEMORY,
    torch_dtype=compute_dtype,
)
tokenizer = AutoTokenizer.from_pretrained(merged_checkpoint_dir)

max_length = get_max_length(model)

preprocessed_dataset = preprocess_dataset(
    tokenizer=tokenizer, max_length=max_length, seed=args.seed, dataset=dataset
)df_test_dataset = convert_dataset(dataset)input_list = df_test_dataset["prompt"].to_list()# # Suppress specific warnings from torch.utils.checkpoint
# with warnings.catch_warnings():
#     warnings.filterwarnings(
#         "ignore", category=UserWarning, module="torch.utils.checkpoint"
#     )