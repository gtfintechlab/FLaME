import requests
import os
import zipfile

base_url = "https://github.com/czyssrs/FinQA/raw/main/dataset/"


filenames = ["dev.json", "test.json", "train.json"]


save_dir = "finqa"
os.makedirs(save_dir, exist_ok=True)


def download_file(file_url, local_path):
    with requests.get(file_url, stream=True) as r, open(local_path, "wb") as f:
        if r.status_code == 200:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
            print(f"{os.path.basename(local_path)} downloaded successfully.")
            return True
        else:
            print(
                f"Failed to download {os.path.basename(local_path)}: Status code {r.status_code}"
            )
            return False


for filename in filenames:
    file_url = base_url + filename
    local_path = os.path.join(save_dir, filename)

    if download_file(file_url, local_path):
        if filename == "train.json":
            zip_path = os.path.join(save_dir, "train.json.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(local_path, arcname=filename)
            print(f"{filename} zipped successfully into train.json.zip")
