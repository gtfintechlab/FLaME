import argparse
import yaml
from datasets import load_dataset
from src.utils.sampling_utils import sample_dataset


def main():
    parser = argparse.ArgumentParser(description="Run inference on a dataset.")
    parser.add_argument("--config", type=str, help="Path to the YAML config file.")
    parser.add_argument(
        "--sample_size", type=int, default=100, help="Number of samples to use."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["random", "head", "tail"],
        default="random",
        help="Sampling method.",
    )

    args = parser.parse_args()

    # Load configuration from YAML if provided
    if args.config:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
            args.sample_size = config.get("sample_size", args.sample_size)
            args.method = config.get("method", args.method)

    # Load dataset
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset to use."
    )

    # Load dataset
    dataset = load_dataset(args.dataset)

    # Sample the dataset
    sampled_data = sample_dataset(dataset["train"], args.sample_size, args.method)

    # Run inference on sampled_data
    print(sampled_data)


if __name__ == "__main__":
    main()
