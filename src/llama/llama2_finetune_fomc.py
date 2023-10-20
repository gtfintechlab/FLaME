class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

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