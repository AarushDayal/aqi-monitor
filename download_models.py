import os
import requests

def download_if_missing(filepath, url):
    if not os.path.exists(filepath):
        print(f"Downloading {filepath}...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filepath}")
