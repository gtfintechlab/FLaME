import zipfile
from pathlib import Path

import pandas as pd


def zip_to_csv(zip_file_path, json_file_name, csv_file_path):
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extract(json_file_name, Path(zip_file_path).parent)
    json_path = Path(zip_file_path).parent / json_file_name
    df = pd.read_json(json_path)
    df.to_csv(csv_file_path, index=False)
