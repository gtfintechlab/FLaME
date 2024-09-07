`glennmatlin[at]gatech[dot]edu`

# Project Repository

This repository is organized into three primary components: Data Management, Inference Pipeline, and Instructions for Running the Code.

---

## 1. Data Management

The `data` folder in this repository contains multiple subfolders, each representing a different dataset. For each dataset, there is a corresponding script named `huggify_{dataset}.py`. These scripts are designed to upload the raw data for each dataset to the [gtfintech lab's repository on Hugging Face](https://huggingface.co/gtfintechlab).

### How to Use the Data Upload Script

For each dataset, you can find its respective script inside its corresponding folder. The script reads raw data files (in CSV format), processes them, and uploads them to Hugging Face using the Hugging Face API.

- Example directory structure:
data/ ├── DatasetA/ │ └── huggify_DatasetA.py ├── DatasetB/ │ └── huggify_DatasetB.py



To upload a dataset, simply run the respective script:


python3 data/{dataset_name}/huggify_{dataset_name}.py


## 2. Inference Pipeline

The main entry point for the inference process is src/together/inference.py. This script serves as the core orchestrator for running inference tasks on different datasets. It manages the API calls, model loading, and task-specific configurations.

Dataset-Specific Inference Scripts
Each dataset has a dedicated inference script located under src/together/{dataset_name}_inference.py. These scripts contain task-specific inference logic and handle the input/output formats for that particular dataset.

src/together/
  ├── DatasetA_inference.py
  ├── DatasetB_inference.py
  ├── prompts.py
  └── inference.py

Prompts
The file src/together/prompts.py holds various zero-shot prompts that are used for each dataset during inference. These prompts guide the model during the prediction phase.

## 3. Running the inference pipeline

To run inference on any dataset using this repository, you can use the following command:

python3 src/together/inference.py --model "{model_name}" --task "{dataset_name}" --api_key "{api_key}" --hf_token "{hf_token}" --max_tokens {max_tokens} --temperature {temperature} --top_p {top_p} --top_k {top_k} --repetition_penalty {repetition_penalty} --prompt_format "{prompt_format}"


Command Options:
--model: The name of the model you want to use for inference (e.g., GPT-3, T5, etc.).
--task: The name of the dataset task for which you are running inference.
--api_key: Your API key for external model services (e.g., OpenAI, etc.).
--hf_token: Your Hugging Face token for accessing models and datasets.
--max_tokens: The maximum number of tokens to generate for each inference.
--temperature: The sampling temperature (controls randomness in predictions).
--top_p: Controls nucleus sampling.
--top_k: Controls top-k sampling.
--repetition_penalty: Penalty for repeated tokens during inference.
--prompt_format: Specify the format of the prompt you want to use (from prompts.py).



