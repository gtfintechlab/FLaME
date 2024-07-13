def get_model_name(model):
    model_dict = {
        "meta-llama/Llama-2-70b-chat-hf": "Llama-2-70b",
        "meta-llama/Llama-2-7b-chat-hf": "Llama-2-7b",
        "meta-llama/Llama-3-70b-chat-hf": "Llama-3-70b",
        "meta-llama/Llama-3-8b-chat-hf": "Llama-3-8b",
    }

    return model_dict.get(model, model)
