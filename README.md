# indabax-2025
Supporting resources for a recommender systems workshop at the [Deep Learning IndabaX South Africa 2025](https://indabax.co.za/).

## Topic
Recommender Systems at Takealot.com

## Presenter
Stefan Dominicus (Senior Machine Learning Software Engineer at Takealot.com)

# Getting Started
To confirm that your environment is correctly configured, let's run though a very basic workflow.

```sh
# Create a virtual environment
VIRTUAL_ENVIRONMENT_NAME=indabax-2025
conda create -n $VIRTUAL_ENVIRONMENT_NAME -y python=3.10
conda activate $VIRTUAL_ENVIRONMENT_NAME

# Install the required dependencies
make install

# Pull the dataset CSV files we'll be working with today
python -m recommender_systems.pull_data
```
