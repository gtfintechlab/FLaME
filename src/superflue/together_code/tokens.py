from typing import List

def tokens(api_model_string: str) -> List[str]:
    """
    Returns the stop tokens for a given Language Model (LLM) based on its API Model String.

    Parameters:
        api_model_string (str): The API Model String of the LLM (e.g., 'Qwen/Qwen1.5-72B-Chat').

    Returns:
        List[str]: A list of stop tokens associated with the specified model.
                   Returns an empty list if the model is not recognized.
    """
    stop_tokens_mapping = {
        # Qwen2, Qwen1.5
        'Qwen/Qwen1.5-72B-Chat': ["<|im_end|>", "<|im_start|>"],
        'Qwen/Qwen1.5-110B-Chat': ["<|im_end|>", "<|im_start|>"],
        'Qwen/Qwen2-72B-Instruct': ["<|im_end|>", "<|im_start|>"],

        # Mixtral, Mistral, Llama2
        'mistralai/Mistral-7B-Instruct-v0.1': ["[/INST]", "</s>"],
        'mistralai/Mistral-7B-Instruct-v0.2': ["[/INST]", "</s>"],
        'mistralai/Mistral-7B-Instruct-v0.3': ["[/INST]", "</s>"],
        'mistralai/Mixtral-8x7B-Instruct-v0.1': ["[/INST]", "</s>"],
        'mistralai/Mixtral-8x22B-Instruct-v0.1': ["[/INST]", "</s>"],
        'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO': ["[/INST]", "</s>"],
        'meta-llama/Llama-2-13b-chat-hf': ["[/INST]", "</s>"],
        'together_ai/meta-llama/Llama-3-70b-chat-hf': ["[/INST]", "</s>"],
        'together_ai/meta-llama/Llama-3-8b-chat-hf': ["[/INST]", "</s>"],

        # Llama3.1, Llama3.2
        'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo': ["<|eot_id|>", "<|eom_id|>"],
        'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo': ["<|eot_id|>", "<|eom_id|>"],
        'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo': ["<|eot_id|>", "<|eom_id|>"],
        'meta-llama/Meta-Llama-3-8B-Instruct-Turbo': ["<|eot_id|>", "<|eom_id|>"],
        'meta-llama/Meta-Llama-3-70B-Instruct-Turbo': ["<|eot_id|>", "<|eom_id|>"],
        'meta-llama/Llama-3-8b-chat-hf': ["<|eot_id|>", "<|eom_id|>"],
        'meta-llama/Llama-3-70b-chat-hf': ["<|eot_id|>", "<|eom_id|>"],
        'NousResearch/Hermes-3-Llama-3.1-405B-Turbo': ["<|eot_id|>", "<|eom_id|>"],

        # Gemma
        'google/gemma-2-27b-it': ["<end_of_turn>", "<eos>"],
        'google/gemma-2-9b-it': ["<end_of_turn>", "<eos>"],
        'google/gemma-2b-it': ["<end_of_turn>", "<eos>"],

        # DeepSeek
        'deepseek-ai/deepseek-llm-67b-chat': ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"],
    }

    return stop_tokens_mapping.get(api_model_string, [])
