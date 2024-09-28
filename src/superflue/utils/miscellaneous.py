import io
import zipfile

import requests


def download_zip_content(url):
    response = requests.get(url)
    return zipfile.ZipFile(io.BytesIO(response.content))
