import requests
import pandas as pd
from multiprocessing import Pool
import os

def download_image(data):
    idx, url = data
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            file_path = f"image_{idx}.jpg"
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Downloaded {file_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def main(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    urls = df.image_url.tolist()

    # Create a directory for images
    if not os.path.exists('data/images'):
        os.makedirs('data/images')
    os.chdir('data/images')

    # Use multiprocessing to download images
    with Pool(15) as p:
        p.map(download_image, enumerate(urls))

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    csv_file = 'shutterstock.csv'  # replace with your CSV file path
    main(csv_file)

