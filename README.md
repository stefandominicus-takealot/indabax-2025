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

# Add a kernel from that environment
DL_ANACONDA_ENV_HOME=$DL_ANACONDA_HOME/envs/$VIRTUAL_ENVIRONMENT_NAME
python -m ipykernel install --prefix $DL_ANACONDA_ENV_HOME --name $VIRTUAL_ENVIRONMENT_NAME --display-name $VIRTUAL_ENVIRONMENT_NAME
rm -rf /opt/conda/envs/$VIRTUAL_ENVIRONMENT_NAME/share/jupyter/kernels/python3

# Pull the dataset CSV files we'll be working with today
python -m recommender_systems.pull_data
```
