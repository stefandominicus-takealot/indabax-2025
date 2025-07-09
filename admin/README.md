# Getting started
1. Make sure the dataset CSV files are ready (details [below](#data)).
2. Create a workbench instance for each participant (details [below](#workbench-instances)).
3. Each participant must open their notebook, and clone the repo
   - Visit https://console.cloud.google.com/vertex-ai/workbench/instances?project=tal-deep-learning-indabax
   - Click `Open JupyterLab` on their instance.
   - Clone the repo:
     - `Git > Clone a Repository`
     - https://github.com/stefandominicus-takealot/indabax-2025
   - Double-click the `indabax-2025` directory in the file browser pane.
4. Open `README.md` and follow the `Getting Started` instructions.
   - Create a new terminal from the launcher in order to run the commands.

## Data

```sh
# Fetch datasets from BigQuery, and persist locally as CSV files
python -m admin.fetch_data

# Push those CSV files (excluding the test set) to GCS
python -m admin.push_data
```

## Workbench Instances
We need to create one instance per participant.

```sh
INSTANCE_NAME='stefan-dominicus'

gcloud \
--project=tal-deep-learning-indabax \
workbench instances create $INSTANCE_NAME \
--location=europe-west1-b \
--machine-type=n1-standard-4 \
--metadata=idle-timeout-seconds=3600 \
--accelerator-core-count=1 \
--accelerator-type=NVIDIA_TESLA_T4 \
--install-gpu-driver \
--network=default \
--subnet=default \
--subnet-region=europe-west1
```

## Leaderboard

```sh
streamlit run admin/leaderboard.py
```
