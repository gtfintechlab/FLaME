import argparse
import logging
import os
import gc
import torch
from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from typing import Optional

def setup_logging(log_file: str = "llama_finetune.log") -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("llama_finetune")
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

def memory_cleanup():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def fine_tune_llama31(
    dataset_path: str,
    model_name: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    max_length: int = 512,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    gradient_clip: float = None,
    fp16: bool = False,
    bf16: bool = False,
) -> Optional[str]:
    """
    Fine-tune a LLaMA 3.1 model on the Economics TestBank dataset.

    Args:
        dataset_path: Path to the dataset with rationalized reasoning
        model_name: Name of the pre-trained LLaMA 3.1 model
        output_dir: Directory to save the fine-tuned model
        epochs: Number of training epochs
        batch_size: Batch size for training
        max_length: Maximum sequence length
        learning_rate: Learning rate for training
        weight_decay: Weight decay for regularization
        warmup_ratio: Ratio of warmup steps
        gradient_clip: Max gradient norm for clipping (None to disable)
        fp16: Enable 16-bit floating point precision
        bf16: Enable bfloat16 precision
    Returns:
        str: Path to the saved model, or None if training fails
    """
    logger = setup_logging()
    logger.info(f"Starting fine-tuning process with model {model_name}")
    
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess dataset
        logger.info("Loading dataset...")
        try:
            df = pd.read_csv(dataset_path)
            required_columns = ["Prompt", "Correct_Answer", "Refined_Rationale"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Dataset missing required columns: {required_columns}")
            
            df = df[required_columns].astype(str)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        # Prepare input-output pairs
        df["input_text"] = df["Prompt"]
        df["target_text"] = df["Correct_Answer"] + "\n\nReasoning:\n" + df["Refined_Rationale"]
        
        # Convert to HF Dataset
        logger.info("Converting to HuggingFace Dataset format...")
        dataset = Dataset.from_pandas(df[["input_text", "target_text"]])

        # Initialize tokenizer
        logger.info("Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

        def tokenize_function(examples):
            return tokenizer(
                examples["input_text"],
                text_target=examples["target_text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )

        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["input_text", "target_text"],
        )

        # Load model
        logger.info("Loading model...")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            logger.info(f"Model loaded successfully. Parameters: {model.num_parameters():,}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            evaluation_strategy="no",  # Disable evaluation for now
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            max_grad_norm=gradient_clip,
            fp16=fp16,
            bf16=bf16,
            logging_steps=10,  # Frequent logging for monitoring
            save_strategy="epoch",
            save_total_limit=2,  # Keep only last 2 checkpoints
            remove_unused_columns=False,
            report_to="none",  # Disable wandb/tensorboard for now
        )

        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )

        # Training
        logger.info("Starting training...")
        try:
            trainer.train()
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        # Save model
        logger.info("Saving model...")
        try:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Model successfully saved to {output_dir}")
            return str(output_dir)
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        return None
    finally:
        memory_cleanup()
        logger.info("Fine-tuning process completed")

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune LLaMA model on custom dataset')
    
    # Required arguments
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to the dataset CSV file')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name or path of the pre-trained model')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save the fine-tuned model')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Training batch size')
    parser.add_argument('--max_length', type=int, default=512,
                      help='Maximum sequence length')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay for regularization')
    parser.add_argument('--warmup_ratio', type=float, default=0.03,
                      help='Ratio of warmup steps')
    
    # Training optimizations
    parser.add_argument('--gradient_clip', type=float, default=None,
                      help='Max gradient norm for clipping (None to disable)')
    parser.add_argument('--fp16', action='store_true',
                      help='Enable mixed precision training with fp16')
    parser.add_argument('--bf16', action='store_true',
                      help='Enable mixed precision training with bf16')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Configure CUDA - using safer defaults
    if torch.cuda.is_available():
        # Set to a more conservative memory split
        torch.cuda.set_per_process_memory_fraction(0.9)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    
    # Example paths are now relative to the project root
    fine_tune_llama31(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_clip=args.gradient_clip,
        fp16=args.fp16,
        bf16=args.bf16,
    )
