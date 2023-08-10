import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

from utils.config import VALID_MODELS
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils.logging import setup_logger

# TODO: Add support to ensure we load from local model cache; perhaps set HF to offline mode?
# import os
# model_cache=os.path.join(
#     os.path.expanduser("~"), f"models_hf/{args.model_id.split('/')[0]}"
# )
# from pathlib import Path
# ROOT_DIRECTORY = Path(__file__).resolve().parent.parent.parent.parent
# CACHE_DIR=str(ROOT_DIRECTORY / ".model_cache")

logger = setup_logger(__name__)

def get_hf_model(args):
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

    if args.model_id not in VALID_MODELS:
        raise ValueError(f"Invalid model '{args.model_id}'")

    model_config = LlamaConfig.from_pretrained(
        args.model_id, use_auth_token=args.hf_auth
    )

    logger.info(
        f"Loading model '{args.model_id}' with quantization '{args.quantization}'"
    )
    # TODO: determine how to create a model param dict that can be passed to the model instead of repeating
    if args.quantization == "default":
        model = LlamaForCausalLM.from_pretrained(
            args.model_id,
            use_auth_token=args.hf_auth,
            trust_remote_code=True,
            config=model_config,
            device_map="auto",
            max_memory=CUDA_MAX_MEMORY,
        )

    elif args.quantization == "bf16":
        model = LlamaForCausalLM.from_pretrained(
            args.model_id,
            use_auth_token=args.hf_auth,
            trust_remote_code=True,
            config=model_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=CUDA_MAX_MEMORY,
        )
    elif args.quantization == "int8":
        model = LlamaForCausalLM.from_pretrained(
            args.model_id,
            use_auth_token=args.hf_auth,
            trust_remote_code=True,
            config=model_config,
            load_in_8bit=True,
            device_map="auto",
            max_memory=CUDA_MAX_MEMORY,
        )
    elif args.quantization == "int4":
        model = LlamaForCausalLM.from_pretrained(
            args.model_id,
            use_auth_token=args.hf_auth,
            trust_remote_code=True,
            config=model_config,
            load_in_4bit=True,
            device_map="auto",
            max_memory=CUDA_MAX_MEMORY,
        )
    else:
        raise ValueError(f"Invalid quantization '{args.quantization}'")
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_id,
        use_auth_token=args.hf_auth,
        legacy=False,
        add_bos_token=True,
        add_eos_token=False,
    )
    return model, tokenizer


def get_token_id(tokenizer: LlamaTokenizer, key: str) -> int:
    """
    What:
        Retrieves the token ID for a given string from a LlamaTokenizer.
    Why:
        When Llama was trained, special tokens were used to mark the beginning and end of instructions and responses.
        We use this function to identify the token IDs for these special tokens.
    Args:
        tokenizer (LlamaTokenizer): the tokenizer
        key (str): the key to convert to a single token
    Raises:
        RuntimeError: if more than one ID was generated
    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(
            f"Expected only a single token for '{key}' but found {token_ids}"
        )
    return token_ids[0]
