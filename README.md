# SuperFLUE (Financial Language Understanding Evaluation)
Corresponding Authors: `glennmatlin[at]gatech[dot]edu` `huzaifa[at]gatech[dot]edu`
***ASSUME ALL THE TESTS ARE OBSOLETE DONT WORRY ABOUT THEM RIGHT NOW***

## Project Setup

### Creating and Activating the Virtual Environment

To create the virtual environment in the project root and install the required packages, follow these steps:

1. **Create the virtual environment**:
    ```sh
    python -m venv .venv
    ```

2. **Activate the virtual environment**:
    - On Windows:
        ```sh
        .\.venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source .venv/bin/activate
        ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

### Installing SuperFLUE

From the root directory you can run `pip install -e .` -- this uses `setup.py` to install SuperFLUE to your activate Python environment.

You can re-install superflue if something goes wrong:
```bash
pip uninstall superflue
pip install -e .
```

(unsure if needed) Clean-up files after install:
```bash
python setup.py clean --all
rm -rf build/ dist/ *.egg-info
find . -name '*.pyc' -delete
find . -name '__pycache__' -delete
```

Test the installation of SuperFLUE worked:
```bash
python
>>> import superflue
>>> print(superflue.__file__)
```

### API keys
To configure your API keys, follow these steps:

1. **Create a `.env` file**:
    - You can create a new `.env` file in the project root directory **OR** copy the provided `.env.sample` file and rename it to `.env`.

2. **Modify the `.env` file**:
    - Open the `.env` file in a text editor.
    - Add your API keys in the following format:
      ```
      API_KEY_NAME=your_api_key_value
      ```
    - Replace `API_KEY_NAME` with the actual name of the API key and `your_api_key_value` with your actual API key.
    - Make sure to add all API keys relevant to the models you want to call.

3. **Save the `.env` file**:
    - Ensure the file is saved in the project root directory.

Example:
```
HUGGINGFACEHUB_API_TOKEN=foo
TOGETHER_API_KEY=bar
OPENAI_API_KEY=buzz
ANTHROPIC_API_KEY=buzz
```

## Project Repository

This repository is organized into three primary components: Data Management, Inference Pipeline, and Instructions for Running the Code.

---

### 1. Data Management

The `data` folder in this repository contains multiple subfolders, each representing a different dataset. For each dataset, there is a corresponding script named `huggify_{dataset}.py`. These scripts are designed to upload the raw data for each dataset to the [gtfintech lab's repository on Hugging Face](https://huggingface.co/gtfintechlab).

#### How to Use the Data Upload Script

For each dataset, you can find its respective script inside its corresponding folder. The script reads raw data files (in CSV/JSON/other formats), processes them, and uploads them to Hugging Face using the Hugging Face API.

- Example directory structure:
```bash
data/ 
    ├── DatasetA/ 
    │ └── huggify_DatasetA.py 
    ├── DatasetB/ 
    │ └── huggify_DatasetB.py
```


To upload a dataset, simply run the respective script:


python3 data/{dataset_name}/huggify_{dataset_name}.py


### 2. Inference Pipeline

The main entry point for the inference process is src/together/inference.py. This script serves as the core orchestrator for running inference tasks on different datasets. It manages the API calls, model loading, and task-specific configurations.

Dataset-Specific Inference Scripts
Each dataset has a dedicated inference script located under src/together/{dataset_name}_inference.py. These scripts contain task-specific inference logic and handle the input/output formats for that particular dataset.

```bash
src/together/
  ├── DatasetA_inference.py
  ├── DatasetB_inference.py
  ├── prompts.py
  └── inference.py
```

Prompts
The file src/together/prompts.py holds various zero-shot prompts that are used for each dataset during inference. These prompts guide the model during the prediction phase.

### 3. Running the inference pipeline

To run inference on any dataset using this repository, you can use the following command:

`python3 src/together/inference.py --model "{model_name}" --task "{dataset_name}" --max_tokens {max_tokens} --temperature {temperature} --top_p {top_p} --top_k {top_k} --repetition_penalty {repetition_penalty} --prompt_format "{prompt_format}"`


#### Command Options:
- `--model`: The name of the model you want to use for inference (e.g., GPT-3, T5, etc.).
- `--task`: The name of the dataset task for which you are running inference.
- `--max_tokens`: The maximum number of tokens to generate for each inference.
- `--temperature`: The sampling temperature (controls randomness in predictions).
- `--top_p`: Controls nucleus sampling.
<!-- - `--top_k`: Controls top-k sampling. -->
- `--repetition_penalty`: Penalty for repeated tokens during inference.
- `--prompt_format`: Specify the format of the prompt you want to use (from `prompts.py`).