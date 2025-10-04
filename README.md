# python-predictive-model

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/libraries-scikit--learn%20%7C%20xarray%20%7C%20pandas-orange.svg)

A regional climate forecasting model using RandomForest trained on historical NASA MERRA-2 data. It leverages latitude and longitude as features to generate location-aware forecasts for temperature, rain probability, and wind for any coordinate within the trained area.



## 📖 Overview

This project provides a complete pipeline for training a regional climate forecasting model. Instead of training a model for a single point, it learns from data across a geographical bounding box (e.g., the Arequipa region) by using latitude and longitude as input features. This allows it to generate predictions for any coordinate within that trained area.

The system uses a suite of four `RandomForest` models to predict different weather aspects, providing a comprehensive forecast.

## 🛠️ Technology Stack

* **Language**: Python
* **Data Access**: `earthaccess` for connecting to the NASA Earthdata repository.
* **Data Manipulation**: `xarray` for handling NetCDF files and `pandas` for data structuring.
* **Machine Learning**: `scikit-learn` for `RandomForest` models.
* **Model Persistence**: `joblib` for saving and loading trained models.

## 🚀 Usage

The project is split into three main scripts: one for training and one for prediction.

### 1. Training the Models

Run the training script from your terminal. This will download all the necessary historical data, process it, train the four models, and save them as `.joblib` files.

**Warning:** This process is data-intensive and will take a very long time to complete for a 20-year period. It is recommended to run it on a machine with a stable internet connection and sufficient processing power.
