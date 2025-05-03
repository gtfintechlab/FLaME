import io
import zipfile

import requests
# TODO: (Glenn) flame.utils.miscellaneous is one function -- it can be refactored elsewhere and deleted


def download_zip_content(url):
    response = requests.get(url)
    return zipfile.ZipFile(io.BytesIO(response.content))
