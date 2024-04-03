# Machine Learning for the Detection of Atmospheric Blocking Events during European Summer

## Thesis

This research was conducted in a bachelor's thesis in 2024 at the Institue for AI in Climate and Environmental Sciences at Karlsruhe Institute for Technology (KIT). The bachelor's thesis is available here TODO.

## Data

The corresponding datasets are taken from the European Centre for Medium-Range Weather Forecasts (ERA5) and the UK Earth System Modelling Project (UKESM). This includes geopotential height at 500hPa and mean sea level pressure. For labels, the ground truth dataset from [Thomas et al](https://doi.org/10.5194/wcd-2-581-2021) is used. This includes ERA5 and UKESM labels. A deeper explanation into the data used and preprocessing necessary is available in the thesis.

### Preprocessing

All preprocessing steps are stored in a Jupyter Notebook located at `bin/transform-data.py`. Individual steps are defined in-depth in the bachelor's thesis. To preprocess a different datasets, change the corresponding variables in the beginning according to your netCDF4 file and meteorological variable.

### Visualizations

Visualizations to control data preprocessing and visualize important aspects are available in `bin/visualize-data.py`. This also includes data visualizations from the bachelor's thesis. To use the visualizations with a custom dataset, update the variables in the beginning according to the setup in your netCDF4 file.

## Setup

### Environment

For dependency management this repository is using [Poetry](https://python-poetry.org/). To get up and running follow the install instructions for Poetry on the official website. After Poetry setup the dependencies can be installed with `poetry install`.

To activate the created environment use `poetry shell`. You can also execute specific commands directly in the environment without activating by using `poetry run`. All slurm jobs are already setup to use the correct Python environment.

### RF-BI (Random Forest Classifier)

RF-BI is based on a random forest classifier fitted directly on observational and model data. All functionality for RF-BI are located in the Jupyter Notebook `model/rf.ipynb`.

#### Fitting

Random forests use a hyperparameter grid search and are fitted using `sklearn`. To fit a new version of RF-BI follow the steps described in `model/rf.ipynb`. For training and search, a setup with at least 10 CPU cores is recommended.

#### Inference

To use RF-BI during inference, record the best hyperparameter combination found during search and fit a singular random forest classifier according to your needs. This is also demonstrated in the Jupyter Notebook. The model can then be deployed to various providers and used in applications.

### CONV-BI (Convolutional Neural Network)

CONV-BI is based on pre-trained convolutional neural networks and fine-tuned on observational and model data.

#### Training

For training CONV-BI utilizes a vast hyperparameter search with [Propulate](https://propulate.readthedocs.io/en/latest/) is conducted. For this search, a cluster setup with four connected A100 40GB GPUs is recommended. For final model training and inference smaller computational capabilities are required. All search functionality can be found in `model/propulate_search.py`, final training and testing on the same dataset is available in `model/model.py` and for transferability between different datasets in `model/train_test.py`. For loading a pre-trained network and fine-tuning on a dataset, functionality is contained in `model/test.py`. For both, Propulate search and model training, slurm jobs are available and used during training. They are stored in `jobs/` and require minor configuration to the output repositories.

#### Inference

During inference, models are directly loaded from the saved PyTorch weights. Important on weight loading is the correct model architecture during training of the weights and inference. All models defined in `models/*.py` accept an optional initialization paramter `weights` to load an already trained network for use during inference or further fine-tuning.