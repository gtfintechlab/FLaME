import argparse

import yaml
from src.together_code.inference import main as inference

def parse_arguments():
    parser = argparse.ArgumentParser(description="SuperFLUE")
    # parser.add_argument("--api_key, required=False", type=str, help="API key to use")
    # parser.add_argument("--hf_token", type=str, help="Hugging Face token to use")
    parser.add_argument("--config", type=str, help="Path to the YAML config file.")
    parser.add_argument(
        "--dataset", type=str, help="Name of the dataset to use."
    )

    # # Glenn: Sampling is not currently working -- the dataframes are loaded in the `inference`
    # # functions for each task which means they need to have this arg info passed down
    # # to have it have an effect.
    # parser.add_argument(
    #     "--sample_size", type=int, default=10, help="Number of samples to use."
    # )
    # parser.add_argument(
    #     "--method",
    #     type=str,
    #     choices=["random", "head", "tail"],
    #     default="random",
    #     help="Sampling method.",
    # )

    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Max tokens to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature to use",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p to use"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Inference batch size",
    )
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="superflue",
        help="Version of the prompt to use",
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load configuration from YAML if provided
    if args.config:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
            # args.sample_size = config.get("sample_size", args.sample_size)
            # args.method = config.get("method", args.method)
            # args.dataset = config.get("dataset", args.dataset)
            
    inference(args)

    # Run inference on sampled_data
    # print(sampled_data)


if __name__ == "__main__":
    main()
