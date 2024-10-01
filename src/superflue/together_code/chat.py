import pytest
from typing import List
from dotenv import load_dotenv
import os
from together import Together
from superflue.utils.logging_utils import setup_logger
from superflue.config import LOG_DIR, LOG_LEVEL
import time

logger = setup_logger(
    name="together_chat", log_file=LOG_DIR / "together_chat.log", level=LOG_LEVEL
)

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

client = Together(api_key=TOGETHER_API_KEY)


def together_chat(prompt, user_message, args):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        response = client.chat.completions.create(
            messages=messages,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop=get_stop_tokens(args.model),
        )
        response_message = (
            response.choices[0].message if response and response.choices else None
        )
        return response_message
    except Exception as e:
        logger.error(f"Error: {e}. Retrying in 10 seconds.")
        time.sleep(5.0)
        raise e


stop_tokens_mapping = {
    # Qwen2, Qwen1.5
    "Qwen/Qwen1.5-72B-Chat": ["<|im_end|>", "<|im_start|>"],
    "Qwen/Qwen1.5-110B-Chat": ["<|im_end|>", "<|im_start|>"],
    "Qwen/Qwen2-72B-Instruct": ["<|im_end|>", "<|im_start|>"],
    # Mixtral, Mistral, Llama2
    "mistralai/Mistral-7B-Instruct-v0.1": ["[/INST]", "</s>"],
    "mistralai/Mistral-7B-Instruct-v0.2": ["[/INST]", "</s>"],
    "mistralai/Mistral-7B-Instruct-v0.3": ["[/INST]", "</s>"],
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ["[/INST]", "</s>"],
    "mistralai/Mixtral-8x22B-Instruct-v0.1": ["[/INST]", "</s>"],
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": ["[/INST]", "</s>"],
    "meta-llama/Llama-2-13b-chat-hf": ["[/INST]", "</s>"],
    # Llama3.1, Llama3.2
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ["<|eot_id|>", "<|eom_id|>"],
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ["<|eot_id|>", "<|eom_id|>"],
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": ["<|eot_id|>", "<|eom_id|>"],
    "meta-llama/Meta-Llama-3-8B-Instruct-Turbo": ["<|eot_id|>", "<|eom_id|>"],
    "meta-llama/Meta-Llama-3-70B-Instruct-Turbo": ["<|eot_id|>", "<|eom_id|>"],
    "meta-llama/Llama-3-8b-chat-hf": ["<|eot_id|>", "<|eom_id|>"],
    "meta-llama/Llama-3-70b-chat-hf": ["<|eot_id|>", "<|eom_id|>"],
    "NousResearch/Hermes-3-Llama-3.1-405B-Turbo": ["<|eot_id|>", "<|eom_id|>"],
    # Gemma
    "google/gemma-2-27b-it": ["<end_of_turn>", "<eos>"],
    "google/gemma-2-9b-it": ["<end_of_turn>", "<eos>"],
    "google/gemma-2b-it": ["<end_of_turn>", "<eos>"],
    # DeepSeek
    "deepseek-ai/deepseek-llm-67b-chat": [
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
    ],
}


def get_stop_tokens(api_model_string: str) -> List[str]:
    """
    Returns the stop tokens for a given Language Model (LLM) based on its API Model String.

    Parameters:
        api_model_string (str): The API Model String of the LLM (e.g., 'Qwen/Qwen1.5-72B-Chat').

    Returns:
        List[str]: A list of stop tokens associated with the specified model.
                   Returns an empty list if the model is not recognized.
    """
    if api_model_string not in stop_tokens_mapping:
        raise KeyError(f"Model '{api_model_string}' not recognized.")
    return stop_tokens_mapping[api_model_string]


@pytest.mark.parametrize(
    "model, expected_tokens",
    [
        ("Qwen/Qwen1.5-72B-Chat", ["<|im_end|>", "<|im_start|>"]),
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", ["[/INST]", "</s>"]),
        ("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", ["<|eot_id|>", "<|eom_id|>"]),
        ("google/gemma-2-27b-it", ["<end_of_turn>", "<eos>"]),
        (
            "deepseek-ai/deepseek-llm-67b-chat",
            ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"],
        ),
    ],
)
def test_valid_models(model, expected_tokens):
    assert get_stop_tokens(model) == expected_tokens


def test_invalid_model():
    with pytest.raises(KeyError):
        get_stop_tokens("invalid/model")


@pytest.mark.parametrize(
    "model", get_stop_tokens.__globals__["stop_tokens_mapping"].keys()
)
def test_token_format(model):
    stop_tokens = get_stop_tokens.__globals__["stop_tokens_mapping"][model]
    assert isinstance(stop_tokens, list)
    assert len(stop_tokens) == 2
    for token in stop_tokens:
        assert isinstance(token, str)
        assert token.strip() != ""


@pytest.mark.parametrize(
    "model", get_stop_tokens.__globals__["stop_tokens_mapping"].keys()
)
def test_model_string_format(model):
    parts = model.split("/")
    assert len(parts) == 2
    assert parts[0].strip() != ""
    assert parts[1].strip() != ""


if __name__ == "__main__":
    pytest.main([__file__])
