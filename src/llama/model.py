from pathlib import Path

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


from src.utils.logging import setup_logger

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent.parent.parent
CACHE_DIR = str(ROOT_DIRECTORY / ".model_cache")

logger = setup_logger(__name__)

VALID_MODELS = ["meta-llama/Llama-2-7b-hf",
                "meta-llama/Llama-2-13b-hf",
                "meta-llama/Llama-2-70b-hf"
                # "meta-llama/Llama-2-7b-chat-hf",
                # "meta-llama/Llama-2-70b-chat-hf",
                # "meta-llama/Llama-2-13b-chat-hf",
                ]


def get_model(args):
    if torch.cuda.is_available():
        CUDA_N_GPUS = torch.cuda.device_count()
        CUDA_MAX_MEMORY = f"{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB"
        CUDA_MAX_MEMORY = {i: CUDA_MAX_MEMORY for i in range(CUDA_N_GPUS)}
        logger.info(
            f"Using k={CUDA_N_GPUS} CUDA GPUs with max memory {CUDA_MAX_MEMORY}"
        )
    else:
        logger.error(f"CUDA Unavailable!")
        raise OSError("CUDA Unavailable!")

    if args.model not in VALID_MODELS:
        raise ValueError(f"Invalid model '{args.model}'")

    logger.info(f"Loading model '{args.model}' with quantization '{args.quantization}'")
    if args.quantization == "default":
        model = LlamaForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            max_memory=CUDA_MAX_MEMORY,
            cache_dir=CACHE_DIR,
        )
    elif args.quantization == "bf16":
        model = LlamaForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=CUDA_MAX_MEMORY,
            cache_dir=CACHE_DIR,
        )
    elif args.quantization == "int8":
        model = LlamaForCausalLM.from_pretrained(
            args.model,
            load_in_8bit=True,
            device_map="auto",
            max_memory=CUDA_MAX_MEMORY,
            cache_dir=CACHE_DIR,
        )
    elif args.quantization == "int4":
        model = LlamaForCausalLM.from_pretrained(
            args.model,
            load_in_4bit=True,
            device_map="auto",
            max_memory=CUDA_MAX_MEMORY,
            cache_dir=CACHE_DIR,
        )
    else:
        raise ValueError(f"Invalid quantization '{args.quantization}'")
    # TODO: check that `padding_side="left"` is needed -- the Llama quickstart did not use it
    tokenizer = LlamaTokenizer.from_pretrained(args.model, padding_side="left")
    return model, tokenizer
