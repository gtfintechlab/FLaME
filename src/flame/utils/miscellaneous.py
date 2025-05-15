import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

# Collection of file and data handling utilities


def download_zip_content(url):
    """Download and create a ZipFile object from a URL.

    Args:
        url: URL to download from

    Returns:
        A ZipFile object containing the downloaded content
    """
    response = requests.get(url)
    return zipfile.ZipFile(io.BytesIO(response.content))


def zip_to_csv(zip_file_path, json_file_name, csv_file_path):
    """Extract a JSON file from a ZIP archive and convert it to CSV.

    Args:
        zip_file_path: Path to the ZIP file
        json_file_name: Name of the JSON file inside the ZIP
        csv_file_path: Output path for the CSV file
    """
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extract(json_file_name, Path(zip_file_path).parent)
    json_path = Path(zip_file_path).parent / json_file_name
    df = pd.read_json(json_path)
    df.to_csv(csv_file_path, index=False)


def remove_think_tokens(file_path, output_path=None):
    """Remove '<think>' tags and their content from model responses in a CSV file.

    Processes model responses in a CSV file by removing everything before and including
    the '</think>' tag, which is used in some models for showing intermediate thinking steps.

    Args:
        file_path (str or Path): Path to the CSV file containing model responses
        output_path (str or Path, optional): Path for the output CSV file.
            If not provided, defaults to the input path with '_no_think' appended
            before the extension.

    Returns:
        Path: Path to the processed output file

    Raises:
        FileNotFoundError: If the input file doesn't exist
        pd.errors.EmptyDataError: If the CSV file is empty or malformed
    """
    # Convert to Path object for easier path manipulation
    file_path = Path(file_path)

    # Validate input file
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # Determine output path if not provided
    if output_path is None:
        output_path = file_path.parent / f"{file_path.stem}_no_think{file_path.suffix}"
    else:
        output_path = Path(output_path)

    # Create parent directory for output if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process the file
    df = pd.read_csv(file_path)

    # Check if 'llm_responses' column exists
    if "llm_responses" not in df.columns:
        raise ValueError(f"Column 'llm_responses' not found in {file_path}")

    # Process responses - safely handle missing </think> tags
    df["llm_responses"] = df["llm_responses"].apply(
        lambda x: x[(x.find("</think>") + 8) :]
        if isinstance(x, str) and "</think>" in x
        else x
    )

    # Save to output file
    df.to_csv(output_path, index=False)

    return output_path
