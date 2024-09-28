import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run script with task_name and quant arguments."
    )
    # TODO: make sure the provided task name is valid (i.e. in the TASK_DATA_MAP)
    parser.add_argument(
        "-t", "--task_name", type=str, required=True, help="Name of the zero-shot task."
    )
    # TODO: make sure the provided quantization is valid (default, bf16, int8, int4)
    parser.add_argument(
        "-q", "--quantization", type=str, required=True, help="Quantization level."
    )
    # TODO: make sure the provided model is valid (databricks/dolly-v2-12b)
    parser.add_argument(
        "-m", "--model_id", type=str, required=True, help="Official name of the model."
    )
    parser.add_argument(
        "-hf", "--hf_auth", type=str, required=True, help="HuggingFace auth token."
    )
    return parser.parse_args()
