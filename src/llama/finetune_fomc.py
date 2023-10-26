#!/usr/bin/env python
# coding: utf-8

# TODO: change dataset to https://huggingface.co/datasets/gtfintechlab/fomc-example-dataset
# TODO: WandB config as inputs, remove wandb logging from conference notebook tensorboard only
# TODO: Remove tokens from py
# HF_AUTH = os.getenv('HF_AUTH_TOKEN')
# if not HF_AUTH:
#     raise ValueError("HF_AUTH_TOKEN is not set in the environment variables.")
# WANDB_API_KEY = os.getenv('WANDB_API_KEY')
# if not WANDB_API_KEY:
#     raise ValueError("WANDB_API_KEY is not set in the environment variables.")

# ====================== IMPORTS ======================
# Standard Libraries
import os
import gc
import logging
from pathlib import Path
from functools import partial
from typing import NamedTuple, List, Type
from IPython.display import display

# Third-Party Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import wandb
import nltk

# import huggingface_hub
# from tqdm.notebook import tqdm
# from sklearn.model_selection import train_test_split

# PyTorch and HuggingFace Libraries
import torch
import bitsandbytes as bnb
import evaluate
from datasets import Dataset, DatasetDict, load_dataset
from trl import SFTTrainer
from transformers import logging as hf_logging
from transformers.trainer_callback import TrainerCallback
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    TrainingArguments,
    logging,
    # DataCollatorForLanguageModeling,
    # LlamaConfig,
    # LlamaForCausalLM,
    # LlamaModel,
    # LlamaTokenizer,
    # TextGenerationPipeline,
    # Trainer,
    # pipeline,
)
from peft import (
    PeftModel,
    AutoPeftModelForCausalLM,
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ====================== HUGGINGFACE ======================
HF_AUTH = "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"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ====================== WEIGHTS AND BIASES ======================
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

# ====================== USER PARAMETERS ======================
organization = "gtfintechlab"
report_to = "tensorboard"
logging_dir = Path.home() / "tensorboard" / "logs"

# ====================== TASK PARAMETERS ======================
task_name = "fomc_communication"
seeds = (5768, 78516, 944601)
seed = seeds[0]

# ====================== MODEL PARAMETERS ======================
model_parameters = "7b"
model_id = f"meta-llama/Llama-2-{model_parameters}-chat-hf"
model_name = model_id.split("/")[-1]

# ====================== PROMPT PARAMETERS ======================
system_prompt = f"""Discard all previous instructions.
Below is an instruction that describes a task.
Write a response that appropriately completes the request.
"""

instruction_prompt = f"""Behave like you are an expert sentence classifier.
Classify the following sentence from FOMC into 'HAWKISH', 'DOVISH', or 'NEUTRAL' class.
Label 'HAWKISH' if it is corresponding to tightening of the monetary policy.
Label 'DOVISH' if it is corresponding to easing of the monetary policy.
Label 'NEUTRAL' if the stance is neutral.
Provide a single label from the choices 'HAWKISH', 'DOVISH', or 'NEUTRAL' then stop generating text.

The sentence:
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

device_map = "auto"  # Automatically determine the device map

save_safetensors = True

# ====================== BITSANDBYTES PARAMETERS ======================
# Activate 4-bit precision base model loading
load_in_4bit = True

# Activate 8-bit precision base model loading
load_in_8bit = False

# Compute dtype for 4-bit base models
bnb_compute_dtype = compute_dtype

# Quantization type (fp4 or nf4)
bnb_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
bnb_use_double_quant = False


def configure_bnb(args):
    """
    Configures BitsAndBytes based on the arguments provided.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_use_double_quant=args.bnb_use_double_quant,
        bnb_8bit_use_double_quant=args.bnb_use_double_quant,
        bnb_4bit_quant_type=args.bnb_quant_type,
        bnb_8bit_quant_type=args.bnb_quant_type,
        bnb_4bit_compute_dtype=args.bnb_compute_dtype,
        bnb_8bit_compute_dtype=args.bnb_compute_dtype,
    )
    return bnb_config


# ====================== TRAININGARGUMENTS PARAMETERS ======================
# Output directory where the model predictions and checkpoints will be stored
args_output_dir = "/fintech_3/20231018/results"
output_dir = Path(args_output_dir) / f"{model_name}_{task_name}"

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

strategy = "steps"
save_strategy = strategy
logging_strategy = strategy
evaluation_strategy = strategy

disable_tqdm = True
predict_with_generate = True


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


# ====================== ARGUMENTS SETUP ======================
class Args(NamedTuple):
    repo_name: str
    task_name: str
    system_prompt: str
    instruction_prompt: str
    seed: int
    model_id: str
    model_name: str
    organization: str
    lora_r: float
    lora_alpha: float
    lora_dropout: float
    max_seq_length: int
    packing: bool
    device_map: str
    load_in_4bit: bool
    load_in_8bit: bool
    bnb_compute_dtype: bool
    bnb_use_double_quant: bool
    bnb_quant_type: str
    output_dir: str
    num_train_epochs: int
    fp16: bool
    bf16: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    max_grad_norm: float
    learning_rate: float
    weight_decay: float
    optim: str
    lr_scheduler_type: str
    max_steps: int
    warmup_ratio: float
    group_by_length: bool
    save_steps: int
    save_strategy: str
    logging_strategy: str
    logging_steps: int
    evaluation_strategy: str
    neftune_noise_alpha: float
    save_safetensors: bool
    load_best_model_at_end: bool
    disable_tqdm: bool
    B_INST: str
    E_INST: str
    B_SYS: str
    E_SYS: str
    BOS: str
    EOS: str
    report_to: str
    logging_dir: str
    predict_with_generate: bool


def setup_args() -> Args:
    args = Args(
        repo_name=repo_name,
        task_name=task_name,
        system_prompt=system_prompt,
        instruction_prompt=instruction_prompt,
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
        bnb_4bit_compute_dtype=bnb_compute_dtype,
        bnb_4bit_use_double_quant=bnb_use_double_quant,
        bnb_4bit_quant_type=bnb_quant_type,
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
        lr_scheduler_type=lr_scheduler_type,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        save_steps=save_steps,
        save_strategy=save_strategy,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        evaluation_strategy=evaluation_strategy,
        neftune_noise_alpha=neftune_noise_alpha,
        save_safetensors=save_safetensors,
        load_best_model_at_end=load_best_model_at_end,
        disable_tqdm=disable_tqdm,
        B_INST=B_INST,
        E_INST=E_INST,
        B_SYS=B_SYS,
        E_SYS=E_SYS,
        BOS=BOS,
        EOS=EOS,
        report_to=report_to,
        logging_dir=logging_dir,
        predict_with_generate=predict_with_generate,
    )

    return args


# =============== SFT LOGGING FUNCTIONS ==================
def log_trainable_parameters(model, logger):
    """
    Logs the number of trainable parameters in the model.

    Parameters:
    - model : torch.nn.Module - The model to log.
    - logger : logging.Logger - Logger to use for logging the info.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(
        f"Trainable params: {trainable_params} || "
        f"All params: {total_params} || "
        f"Trainable%: {100 * trainable_params / total_params}"
    )


def log_dtypes(model, logger):
    """
    Logs the data types of the model parameters.

    Parameters:
    - model : torch.nn.Module - The model to log.
    - logger : logging.Logger - Logger to use for logging the info.
    """
    dtypes = {}

    for p in model.parameters():
        dtype = p.dtype
        dtypes[dtype] = dtypes.get(dtype, 0) + p.numel()

    total = sum(dtypes.values())

    for dtype, count in dtypes.items():
        logger.info(f"{dtype}: {count} ({100 * count / total:.2f}%)")


def log_and_save_info(model, logger, args):
    """
    Log information and save it for further analysis.
    """
    info_data = []

    logger.debug("Getting the model's memory footprint...")
    memory_footprint = model.get_memory_footprint()
    info_data.append(["Memory Footprint", memory_footprint])

    logger.debug(f"Model Dtypes before preparing for kbit training ...")
    dtypes_after = log_dtypes(
        model, logger
    )  # Assuming log_dtypes returns relevant data
    info_data.append(["Dtypes Before KBit Prep", dtypes_after])

    logger.debug("Using the prepare_model_for_kbit_training method from PEFT...")
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )

    logger.debug(f"Model Dtypes after preparing for kbit training ...")
    dtypes_after = log_dtypes(
        model, logger
    )  # Assuming log_dtypes returns relevant data
    info_data.append(["Dtypes After KBit Prep", dtypes_after])

    logger.debug("Get module names for the linear layers where we add LORA adapters...")
    layers_for_adapters = find_all_linear_names(model)
    info_data.append(["Layers for Adapters", layers_for_adapters])

    logger.info("Create PEFT config for these modules and wrap the model to PEFT...")
    peft_config = create_peft_config(args, layers_for_adapters)

    logger.info(f"Model Dtypes before applying PEFT config ...")
    dtypes_before = log_dtypes(model, logger)
    info_data.append(["Dtypes Before PEFT Config", dtypes_before])

    model = get_peft_model(model, peft_config)

    logger.info(f"Model Dtypes after applying PEFT config ...")
    dtypes_after_peft = log_dtypes(model, logger)
    info_data.append(["Dtypes After PEFT Config", dtypes_after_peft])

    logger.info("Information about the percentage of trainable parameters...")
    trainable_parameters = log_trainable_parameters(model, logger)
    info_data.append(["Trainable Parameters", trainable_parameters])

    # Convert the info_data list into a pandas DataFrame and save it
    df = pd.DataFrame(info_data, columns=["Info", "Value"])
    df.to_csv("model_info.csv", index=False)

    return model


def merge_evaluation_results(
    baseline_results: dict, final_results: dict
) -> pd.DataFrame:
    """
    Merge evaluation results for comparison.

    Parameters:
    baseline_results : dict - The baseline evaluation results.
    final_results : dict - The fine-tuned evaluation results.

    Returns:
    pd.DataFrame - A DataFrame containing merged results.
    """
    all_metrics = set(baseline_results.keys()).union(final_results.keys())
    data = {
        "Metric": list(all_metrics),
        "Baseline": [baseline_results.get(metric, None) for metric in all_metrics],
        "After Fine-tuning": [
            final_results.get(metric, None) for metric in all_metrics
        ],
    }

    return pd.DataFrame(data)


# ========== DATA SET PROCESSING FUNCTIONS ==========
def decode_predictions(predictions, labels, tokenizer):
    """
    Decode predictions and labels from token IDs to strings.
    """
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    return decoded_preds, decoded_labels


def get_max_length(model: Type[torch.nn.Module]) -> int:
    """
    Get the maximum length of position embeddings in the model.

    Parameters:
    - model : torch.nn.Module - The model to inspect

    Returns:
    - int - Maximum length of position embeddings
    """
    conf = model.config
    max_length = None

    # Checking various attributes to determine max length
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(conf, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break

    # Defaulting to 1024 if no length attribute is found
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")

    return max_length


def preprocess_batch(
    batch,
    args: Args,
    tokenizer,
    max_seq_length,
    context_field="sentence",
    response_field="label_decoded",
):
    """
    Creates formatted prompts and tokenizes in batch mode.

    Parameters:
    - batch: dict - Batch containing columns as lists.
    - args: Args - Arguments needed for formatting.
    - tokenizer: AutoTokenizer - Tokenizer for the model.
    - max_seq_length: int - Maximum sequence length for tokenization.
    - context_field: str - The key for context text in the batch.
    - response_field: str - The key for response text in the batch.
    """

    instruction_prompt = args.instruction_prompt
    system_prompt = args.system_prompt

    # Validating the necessary components
    if not instruction_prompt.strip() or not system_prompt.strip():
        raise ValueError("Instruction and system prompts must be non-empty strings.")

    # Check each element of the context_field and response_field
    if not all(item.strip() for item in batch[context_field]) or not all(
        item.strip() for item in batch[response_field]
    ):
        raise ValueError("Fields must be non-empty strings.")

    # Creating the formatted prompt for each sample in the batch
    batch["text"] = [
        args.B_INST
        + args.B_SYS
        + system_prompt
        + args.E_SYS
        + instruction_prompt
        + context
        + args.E_INST
        for context in batch[context_field]
    ]

    # Tokenizing the batch
    return tokenizer(batch["text"], max_length=max_seq_length, truncation=True)


def preprocess_dataset(
    args: Args, tokenizer: AutoTokenizer, max_seq_length: int, dataset: Dataset
):
    """
    Format & tokenize the dataset for training.

    Parameters:
    - args: Args - Arguments needed for formatting.
    - tokenizer: AutoTokenizer - Tokenizer for the model.
    - max_seq_length: int - Maximum sequence length for tokenization.
    - dataset: Dataset - Dataset to preprocess.
    """

    # Applying the preprocessing function to each batch of the dataset
    dataset = dataset.map(
        partial(
            preprocess_batch,
            args=args,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        ),
        batched=True,
    )

    # Further processing steps if necessary (e.g., filtering, shuffling)
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_seq_length)
    dataset = dataset.shuffle(seed=args.seed)

    return dataset


def load_and_preprocess_dataset(args, logger, tokenizer, max_seq_length, split: str):
    """
    Load and preprocess datasets based on the split specified (train/test).
    """
    logger.info(f"Loading {split} dataset...")
    dataset = load_dataset(f"{args.organization}/{args.task_name}", str(args.seed))[
        split
    ]

    logger.info(f"Preprocessing {split} dataset...")
    preprocessed_dataset = preprocess_dataset(
        args=args,
        dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    return preprocessed_dataset


def split_dataset(train_dataset, train_ratio=0.7, seed=42):
    """
    Split a Hugging Face dataset into training and validation sets with a given ratio.

    Parameters:
    - train_dataset: Hugging Face dataset to split
    - train_ratio: Ratio of data to keep in the training set
    - seed: Seed for reproducibility

    Returns:
    - train_set: Training dataset
    - val_set: Validation dataset
    """
    # Ensuring the ratios are valid
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("Train ratio must be between 0 and 1")

    val_ratio = 1 - train_ratio

    # Splitting the dataset
    datasets = train_dataset.train_test_split(test_size=val_ratio, seed=seed)
    train_set = datasets["train"]
    val_set = datasets["test"]

    return train_set, val_set


# ======= PEFT HELPER FUNCTIONS ===========
class PeftSavingCallback(TrainerCallback):
    """
    A callback to save the PEFT adapters during the model training.
    """

    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


def find_all_linear_names(model: Type[torch.nn.Module], bits: int) -> List[str]:
    """
    Find names of all linear layers in the model based on the number of bits specified.

    Parameters:
    - model : torch.nn.Module - The model to inspect
    - bits : int - The number of bits to select the appropriate linear layer class

    Returns:
    - List[str] - List of linear layer names
    """

    # Selecting the appropriate class based on the number of bits
    if bits == 4:
        cls = bnb.nn.Linear4bit
    elif bits == 8:
        cls = bnb.nn.Linear8bitLt
    else:
        cls = torch.nn.Linear

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # Removing 'lm_head' if exists (specific to 16-bit scenarios)
    lora_module_names.discard("lm_head")

    return list(lora_module_names)


def create_peft_config(args: Args, modules: List[str]) -> LoraConfig:
    """
    Create PEFT configuration for LoRA.

    Parameters:
    - args : Args - The arguments containing LoRA parameters
    - modules : List[str] - List of module names

    Returns:
    - LoraConfig - Configuration object for PEFT
    """
    return LoraConfig(
        target_modules=modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# ============== METRICS FUNCTIONS =================


def compute_metrics(eval_pred, tokenizer, metric):
    """
    Compute custom metrics like ROUGE for the given predictions and labels.
    """
    predictions, labels = eval_pred
    decoded_preds, decoded_labels = decode_predictions(predictions, labels, tokenizer)

    # TODO: REVIEW THIS CODE ... SHOULD I EVEN USE NLTK??
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        use_aggregator=True,
    )
    result = {key: value * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def metric_computer(tokenizer):
    """
    Load and compute custom metrics like BLEU and ROUGE.
    """
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    def compute(p):
        predictions, references = p
        pred_ids = np.argmax(p.predictions, axis=2)

        pred_texts = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids
        ]
        label_texts = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in p.label_ids
        ]

        bleu_score = bleu_metric.compute(predictions=pred_texts, references=label_texts)
        rouge_score = rouge_metric.compute(
            predictions=pred_texts, references=label_texts
        )

        return {"bleu": bleu_score, "rouge": rouge_score}

    return compute


# ========== TRAINING FUNCTIONS ===============
def configure_tokenizer(args):
    """
    Configures the tokenizer based on the provided arguments.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=False)
    tokenizer.pad_token = args.EOS

    return tokenizer


def configure_model(args, logger):
    """
    Applies further configurations to the model based on the arguments provided.
    """
    logger.debug("Creating BitsAndBytesConfig ...")
    bnb_config = configure_bnb(args)

    logger.debug("Creating ModelforCausalLM ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        device_map=args.device_map,
        max_memory=CUDA_MAX_MEMORY,
        torch_dtype=compute_dtype,
        quantization_config=bnb_config,
        trust_remote_code=False,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = log_and_save_info(model, logger, args)
    return model


def setup_training_arguments(args):
    """
    Configures and returns the TrainingArguments based on the provided arguments.
    """
    # Directory setup for outputs
    output_dir = setup_output_directory(
        args.output_dir
    )  # Assuming a function for directory setup

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        fp16=args.fp16,
        bf16=args.bf16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
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
        group_by_length=args.group_by_length,
    )
    return training_arguments


def setup_trainer(
    args, model, tokenizer, peft_config, train_dataset, test_dataset, training_arguments
):
    """
    Configures and returns the trainer based on the provided arguments and datasets.
    """
    callbacks = [PeftSavingCallback()]
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        packing=args.packing,
        max_seq_length=max_seq_length,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        callbacks=callbacks,
        dataset_text_field="text",
        neftune_noise_alpha=args.neftune_noise_alpha,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.predict_with_generate = args.predict_with_generate
    return trainer


def execute_training_and_evaluation(trainer, args, logger):
    """
    Executes the training and evaluation process based on the configured trainer and arguments.
    """
    logger.debug(
        "Evaluating the baseline performance of the model before fine-tuning..."
    )
    baseline_results = trainer.evaluate()
    logger.info(f"Baseline evaluation results: {baseline_results}")

    # TODO: improve the try/except blocks
    logger.info("Running trainer.train() ...")
    try:
        trainer.train()
    except Exception as e:
        logger.info("training block failed")
        logger.error(e)
        raise e

    if args.report_to == "wandb":
        wandb.finish()

    logger.info("trainer.evaluate() ...")
    try:
        metrics = trainer.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
    except Exception as e:
        logger.info("metrics block failed")
        logger.error(e)
        raise e

    final_results = trainer.evaluate()
    trainer.save_state()
    logger.info(f"Final evaluation results: {final_results}")
    results_df = merge_evaluation_results(baseline_results, final_results)
    return results_df


def train(args, logger):
    logger.info("Starting Supervised Fine Tuning...")

    # Tokenizer setup and configuration
    logger.debug("Creating the Tokenizer...")
    tokenizer = configure_tokenizer(args)

    # Metrics setup and configuration
    logger.debug("Creating Metrics...")
    compute_metrics_function = metric_computer(tokenizer)
    logger.info(compute_metrics_function)

    model = configure_model(args, logger)
    max_seq_length = get_max_length(
        model
    )  # Assuming function get_max_length is pre-defined

    # Loading and preprocessing datasets
    logger.debug("Loading and preprocessing train dataset...")
    train_dataset = load_and_preprocess_dataset(
        args, logger, tokenizer, max_seq_length, "train"
    )
    train_set, val_set = split_dataset(train_dataset, train_ratio=0.7, seed=args.seed)

    # TrainingArguments setup
    logger.info("Creating TrainingArguments ...")
    training_arguments = setup_training_arguments(args)

    # Trainer setup
    logger.info("Creating SFTTrainer ...")
    trainer = setup_trainer(args, model, train_set, val_set, training_arguments)

    # Training and Evaluation
    results_df = execute_training_and_evaluation(trainer, args, logger)
    display(results_df)

    # Saving final model and tokenizer states
    model = trainer.model
    save_model_and_tokenizer(model, tokenizer, output_dir)


# ======== EVALUATION FUNCTIONS ===============


def generate(model=None, tokenizer=None, dataset=None):
    temperature = 0.0  # [0.0, 1.0]; 0.0 means greedy sampling
    do_sample = False
    max_new_tokens = 256
    top_k = 10
    top_p = 0.92
    repetition_penalty = 1.0  # 1.0 means no penalty
    num_return_sequences = 1  # Only generate one response
    num_beams = 1

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
    return output.split("[/INST]")[-1].strip()


# TODO: INCORPORATE EXTRACT LABEL
def extract_label(text_output, E_INST="[/INST]"):
    # Find the 'end of instruction' token and remove text before it
    response_pos = text_output.find(E_INST)
    generated_text = text_output[response_pos + len(E_INST) :].strip()
    # Convert the string to lowercase for case-insensitive search
    text = text_output.lower()

    # Define the substring options
    substrings = ["label: positive", "label: negative", "label: neutral"]

    # Iterate over the substrings and find the matching label
    for i, substring in enumerate(substrings):
        if substring in text:
            return i

    # If none of the substrings are found, return -1
    return -1


# TODO: INCORPORATE COMPUTE METRICS
def compute_metrics(files, outputs_directory):
    acc_list = []
    f1_list = []
    missing_perc_list = []

    for file in files:
        df = pd.read_csv(outputs_directory / file)

        # Make sure the 'Label:' was provided in all generated text
        if all(df["text_output"].str.contains("Label:")):
            pass
        else:
            raise ValueError("not all responses contain the substring 'Label:'")

        # Decode the predicted label
        df["generated_label"] = df["text_output"].apply(extract_label)

        # Calculate metrics
        acc_list.append(accuracy_score(df["true_label"], df["generated_label"]))
        f1_list.append(
            f1_score(df["true_label"], df["generated_label"], average="weighted")
        )
        missing_perc_list.append(
            (len(df[df["generated_label"] == -1]) / df.shape[0]) * 100.0
        )

    return acc_list, f1_list, missing_perc_list


def evaluate_results():
    # TODO: RESULTS CODE BELOW NEEDS TO BE INCORPORATED
    # results = {}
    # for model_name in model_names:
    #     results[model_name] = {}
    #     for quantization in quantizations:
    #         # Define output directory
    #         LLM_OUTPUTS_DIRECTORY = (
    #             ROOT_DIRECTORY
    #             / "data"
    #             / task_name
    #             / "llm_prompt_outputs"
    #             / quantization
    #         )
    #         # Filter out relevant files
    #         files = [
    #             f.stem
    #             for f in LLM_OUTPUTS_DIRECTORY.iterdir()
    #             if model_name in f.name and f.suffix == ".csv"
    #         ]
    #         results[model_name][quantization] = files
    # acc_list, f1_list, missing_perc_list = compute_metrics(files, LLM_OUTPUTS_DIRECTORY)
    #
    # # Print results
    # print("f1 score mean: ", format(np.mean(f1_list), ".4f"))
    # print("f1 score std: ", format(np.std(f1_list), ".4f"))
    # print(
    #     "Percentage of cases when didn't follow instruction: ",
    #     format(np.mean(missing_perc_list), ".4f"),
    #     "\n",
    # )
    pass


# ====== UTILS =======


def save_model_and_tokenizer(model, tokenizer, model_dir):
    """
    Save the model and tokenizer in the trainer to the specified directory.

    Parameters:
    - trainer
        The trainer object containing the model.
    - tokenizer : PreTrainedTokenizer
        The tokenizer to be saved.
    - model_dir : str
        The directory where the model and tokenizer will be saved.
    """

    try:
        # Save model
        model.save_pretrained(model_dir)
        print(f"Model saved to {model_dir}")

        # Save tokenizer
        tokenizer.save_pretrained(model_dir)
        print(f"Tokenizer saved to {model_dir}")

    except Exception as e:
        print(f"An error occurred while saving the model and tokenizer: {e}")


def memory_cleanup():
    # Empty VRAM
    if "trainer" in locals() or "trainer" in globals():
        del trainer
    if "model" in locals() or "model" in globals():
        del model
    if "pipe" in locals() or "pipe" in globals():
        del pipe
    torch.cuda.empty_cache()
    gc.collect()
    gc.collect()


def load_models(args, logger):
    compute_dtype = args.bnb_compute_dtype

    # Load the foundation model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map=args.device_map,
        max_memory=CUDA_MAX_MEMORY,
        torch_dtype=compute_dtype,
    )
    log_dtypes(base_model, logger)

    bnb_config = configure_bnb(
        args
    )  # Assuming a function configure_bnb exists to set up bnb_config

    # Load the fine-tuned model
    logger.debug("Creating BitsAndBytesConfig ...")
    bnb_config = configure_bnb(args)
    new_model = AutoPeftModelForCausalLM.from_pretrained(
        args.output_dir / "final_checkpoint",
        device_map=args.device_map,
        max_memory=CUDA_MAX_MEMORY,
        torch_dtype=compute_dtype,
        quantization_config=bnb_config,
    )

    log_dtypes(new_model, logger)

    return base_model, new_model


def merge_models(base_model, new_model, logger):
    # Merge the LoRa layers into the base model for standalone use
    peft_model = PeftModel.from_pretrained(base_model, new_model)
    peft_model.merge_and_unload()
    log_dtypes(peft_model, logger)

    return peft_model


def save_and_push(args, peft_model):
    # Save inference
    merged_checkpoint_dir = args.output_dir / "final_merged_checkpoint"
    peft_model.save_pretrained(merged_checkpoint_dir, safe_serialization=True)

    # Load and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.output_dir / "final_checkpoint", pad_token=EOS
    )
    tokenizer.save_pretrained(merged_checkpoint_dir)

    # Push model and tokenizer to hub
    peft_model.push_to_hub(args.repo_name, private=True, use_temp_dir=True)
    tokenizer.push_to_hub(args.repo_name, private=True, use_temp_dir=True)


def setup_output_directory(output_dir_path):
    """
    Sets up the output directory for saving model checkpoints and other outputs.
    """
    output_dir = output_dir_path / "final_checkpoint"
    output_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
    return output_dir


# ========= main ===========
def main():
    args = setup_args()
    logger = setup_logging()
    logger.info(f"Using k={CUDA_N_GPUS} CUDA GPUs with max memory {CUDA_MAX_MEMORY}")

    # if notebook: get_ipython().run_line_magic('tensorboard', '--logdir logs')

    try:
        train(args, logger)
    except Exception as e:
        logger.error(e)
        memory_cleanup()

    base_model, new_model = load_models(args, logger)
    peft_model = merge_models(base_model, new_model, logger)
    save_and_push(args, peft_model)
    max_seq_length = get_max_length(peft_model)

    logger.info("Loading and preprocessing test dataset...")
    logger.debug("Creating Tokenizer...")
    tokenizer = configure_tokenizer(args)
    logger.debug("Creating Test Dataset...")
    test_set = load_and_preprocess_dataset(
        args, logger, tokenizer, max_seq_length, "test"
    )

    # TODO: holdout evaluation
    output_list = []
    for i in range(len(test_set)):
        output_list.append(
            generate(model=peft_model, tokenizer=tokenizer, dataset=test_set)
        )
    output_list.replace("</s>", "")
    return output_list

if __name__ == "__main__":
    main()