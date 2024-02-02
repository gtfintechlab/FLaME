import requests
import zipfile
import os

data_zip_url = 'https://github.com/czyssrs/ConvFinQA/raw/main/data.zip'

save_dir = 'convfinqa'
os.makedirs(save_dir, exist_ok=True)

zip_file_path = os.path.join(save_dir, 'data.zip')

with requests.get(data_zip_url) as response, open(zip_file_path, 'wb') as f:
    if response.status_code == 200:
        f.write(response.content)
        with zipfile.ZipFile(f.name, 'r') as zip_ref:
            for file in ['dev.csv', 'train.json']:
                if file in zip_ref.namelist():
                    zip_ref.extract(file, save_dir)
            train_json_path = os.path.join(save_dir, 'train.json')
            with zipfile.ZipFile(os.path.join(save_dir, 'train.json.zip'), 'w') as train_zip:
                train_zip.write(train_json_path, 'train.json')
            os.remove(train_json_path)
    else:
        print(f'Failed to download file: Status code {response.status_code}')


os.remove(zip_file_path)
