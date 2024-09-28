from datasets import load_dataset
from huggingface_hub import login


def load_data(HF_TOKEN, NAME, SEED):

    # log in to hugging face
    login(HF_TOKEN)
    # check if the dataset has seeds
    if SEED is not None:
        dataset = load_dataset(NAME, f"{SEED}", token=HF_TOKEN)
    else:
        dataset = load_dataset(NAME, token=HF_TOKEN)
    # return the dataset object
    return dataset
