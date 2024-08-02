import argparse
import os

from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and cache a model from the HuggingFace model hub." ""
    )
    # TODO: make sure the provided model is in VALID_MODELS
    parser.add_argument(
        "-m", "--model_id", type=str, required=True, help="Official name of the model."
    )
    parser.add_argument(
        "-hf", "--hf_auth", type=str, required=True, help="HuggingFace auth token."
    )
    return parser.parse_args()


def main(args):
    # Home directory
    home = os.path.expanduser("~")

    # Download each model to a separate folder
    cache_dir = os.path.join(home, f"models_hf/{args.model_id}")
    # Create the directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Download and cache the tokenizer and model
    model_config = LlamaConfig.from_pretrained(
        args.model_id, use_auth_token=args.hf_auth
    )
    print(f"Downloading and caching the model {args.model_id} to {cache_dir} ...")
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_id,
        use_auth_token=args.hf_auth,
        trust_remote_code=True,
        config=model_config,
    )
    model = LlamaForCausalLM.from_pretrained(
        args.model_id,
        use_auth_token=args.hf_auth,
        trust_remote_code=True,
        config=model_config,
    )
    print(f"Model {args.model_id} downloaded and cached successfully!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
