"""
Pull customers, products, and reviews CSV files to GCS.

```sh
python -m recommender_systems.pull_data
```
"""

from pathlib import Path

from google.cloud.storage import Client

if __name__ == "__main__":
    data_path = Path(__file__).parent / "data"

    files = [
        "customers.csv",
        "products.csv",
        "reviews.csv",
        Path("reviews") / "train.csv",
        Path("reviews") / "validation.csv",
    ]

    client = Client(project="tal-deep-learning-indabax")
    bucket = client.bucket("tal-deep-learning-indabax-data")

    for file in files:
        blob = bucket.blob(str(file))
        filename = data_path / file
        filename.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(filename))
        print(f"File {file} downloaded to {data_path / file}.")

    print("All files downloaded successfully.")
