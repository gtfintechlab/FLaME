def get_model_name(model):
    model_dict = {
        "meta-llama/Llama-3-70b-chat-hf": "Llama-3-70b",
    }

    return model_dict.get(model, model)