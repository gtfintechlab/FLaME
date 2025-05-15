#!/usr/bin/env python3
"""
Command-line tool to remove <think> tags from model responses in CSV files.

This script is now a wrapper around the utility function in flame.utils.miscellaneous.
"""

import argparse
import glob
from pathlib import Path

from flame.utils.miscellaneous import remove_think_tokens


def main():
    """Process command line arguments and execute the remove_think_tokens function."""
    parser = argparse.ArgumentParser(
        description="Remove <think> tags from model responses in CSV files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Path(s) to CSV file(s) containing model responses. Supports glob patterns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory for processed files. Default: same as input files.",
    )

    args = parser.parse_args()

    # Expand glob patterns and process each file
    processed_files = []
    for pattern in args.paths:
        for file_path in glob.glob(pattern):
            file_path = Path(file_path)
            if args.output_dir:
                output_path = (
                    args.output_dir / f"{file_path.stem}_no_think{file_path.suffix}"
                )
            else:
                output_path = None  # Let the function determine the default

            try:
                output = remove_think_tokens(file_path, output_path)
                print(f"Processed: {file_path} -> {output}")
                processed_files.append(output)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"Successfully processed {len(processed_files)} file(s)")


if __name__ == "__main__":
    # Usage examples:
    # python -m flame.code.remove_think_tokens "results/**/*DeepSeek*.csv"
    # python -m flame.code.remove_think_tokens results/finer/finer_together_ai/deepseek-ai/DeepSeek-r1_09_02_2025.csv
    main()
