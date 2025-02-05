# Time-Series-Forecasting-Rainfall-Prediction
South Africa Rainfall Prediction

## Project Overview
This project aims to forecast rainfall in South Africa for the remainder of 2025 using historical rainfall data from 1981 to 2025. Accurate rainfall prediction is crucial for water resource management, agriculture, and disaster preparedness in the region.

## Dataset
The dataset used in this project contains historical rainfall measurements for South Africa from January 1981 to January 2025. The data is sourced from a reliable weather monitoring organization.

## Methodology
The project follows these key steps:

### Data Preprocessing:

Load the dataset and parse the date column.

Handle missing values by dropping rows with missing data.

### Data Exploration:

Explore the dataset using descriptive statistics.

Visualize the data with line plots and box plots to identify trends and outliers.

### Stationarity Check:

Perform the Augmented Dickey-Fuller (ADF) test to check for stationarity in the data.

### Data Splitting:

Split the dataset into training (80%) and testing (20%) sets for model evaluation.

### Model Selection and Training:

Use ARIMA (AutoRegressive Integrated Moving Average) model to forecast rainfall.

Perform a grid search to select the best ARIMA parameters based on the AIC (Akaike Information Criterion).

### Model Evaluation:

Evaluate the model's performance using Mean Squared Error (MSE) on the test set.

### Forecasting:

Retrain the ARIMA model on the entire dataset and forecast rainfall for the next 30 days.

Visualize the forecast along with the historical data from 2020 onwards.

### Model Deployment:

Save the trained ARIMA model for future use.

Save the forecasted rainfall data to an Excel file.

## Results
The ARIMA model provides a forecast for the rainfall in South Africa for the remainder of 2025. The forecasted rainfall data is saved to an Excel file, and visualization is created to illustrate the forecast data.

## Value of the project
This project demonstrates the use of time series forecasting techniques to predict rainfall in South Africa. The forecasted rainfall data can be valuable for various stakeholders, including farmers, policymakers, and disaster management authorities.

# Future Task
Comparing actual rainfall in South Africa to the forecasted rainfall
