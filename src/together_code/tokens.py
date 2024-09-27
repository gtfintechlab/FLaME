tokens_map = {"meta-llama/Llama-2-7b-chat-hf": ["<human>", "\n\n"]}


def tokens(model_name):
    return tokens_map.get(model_name, [])
