"""
Write customers, products, and reviews CSV files to GCS.

```sh
python -m admin.push_data
```
"""

from pathlib import Path

from google.cloud.storage import Client

if __name__ == "__main__":
    data_path = Path(__file__).parent / "data"

    files = [
        data_path / "customers.csv",
        data_path / "products.csv",
        data_path / "reviews.csv",
        data_path / "reviews" / "train.csv",
        data_path / "reviews" / "validation.csv",
        # NOTE: Deliberately not uploading `test.csv`.
    ]

    client = Client(project="tal-deep-learning-indabax")
    bucket = client.bucket("tal-deep-learning-indabax-data")

    for file in files:
        blob = bucket.blob(str(file.relative_to(data_path)))
        blob.upload_from_filename(file)
        print(f"File {file} uploaded to {blob.name}.")

    print("All files uploaded successfully.")
