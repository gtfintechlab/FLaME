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
