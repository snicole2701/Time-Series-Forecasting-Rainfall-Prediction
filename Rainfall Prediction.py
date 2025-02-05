import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import itertools

# Load dataset
data = pd.read_csv('South Africa Rainfall .csv')
data['Date '] = pd.to_datetime(data['Date '])
data.set_index('Date ', inplace=True)

# Handle missing values
data = data.dropna()

# Data Exploration
print(data.describe())
data.plot()
plt.title('Rainfall Over Time')
plt.xlabel('Date')
plt.ylabel('Rainfall in mm')
plt.show()

# Box plot
data.boxplot(column='Rainfall in mm')
plt.title('Boxplot of Rainfall Measurements')
plt.show()

# Check for stationarity
result = adfuller(data['Rainfall in mm'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
for key, value in result[4].items():
    print('Critical Values:')
    print(f'{key}, {value}')

# Data Split
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size]['Rainfall in mm'], data.iloc[train_size:]['Rainfall in mm']

# Model Selection and Training with ARIMA
# Grid Search for ARIMA parameters
p = d = q = range(0, 5)
pdq = list(itertools.product(p, d, q))
best_aic = np.inf
best_pdq = None
error_params = []

for param in pdq:
    try:
        temp_model = ARIMA(train, order=param)
        temp_model_fit = temp_model.fit()
        if temp_model_fit.aic < best_aic:
            best_aic = temp_model_fit.aic
            best_pdq = param
    except Exception as e:
        error_params.append((param, str(e)))
        continue

print('Best ARIMA parameters:', best_pdq)

# Model Training
model = ARIMA(train, order=best_pdq)
model_fit = model.fit()

# Model Evaluation
forecast = model_fit.forecast(steps=len(test))
mse = mean_squared_error(test, forecast)
print('Test MSE:', mse)

# Forecasting
model = ARIMA(data['Rainfall in mm'], order=best_pdq)
model_fit = model.fit()
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)

# Visualization: Filter data from 2020 onwards and add the forecast
data_filtered = data['2020':]

# Create a date range for the forecast
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=10), periods=forecast_steps, freq='10D')
forecast_series = pd.Series(forecast, index=forecast_index)


# Plot with larger figure size
plt.figure(figsize=(12, 6))
plt.plot(data_filtered['Rainfall in mm'], label='Actual', color='blue')
plt.plot(forecast_series, label='Forecast', color='red')
plt.legend()
plt.title('Rainfall Forecast from 2020 Onwards')
plt.xlabel('Date')
plt.ylabel('Rainfall in mm')
plt.tight_layout()
plt.show()

# Deployment 
model_fit.save('arima_model.pkl')
print('Model saved for deployment.')

# Save forecast to new spreadsheet
# Create a DataFrame for the forecast
forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecasted Rainfall in mm': forecast})

# Save the DataFrame to a new Excel file
forecast_df.to_excel('Rainfall_Forecast.xlsx', index=False)
print('Forecast saved to Rainfall_Forecast.xlsx')

# Load forecast dataset
forecast_data = pd.read_csv('Rainfall_Forecast.csv')
forecast_data['Date'] = pd.to_datetime(forecast_data['Date'])
forecast_data.set_index('Date', inplace=True)

forecast_data.head()

# Plot forecast data
forecast_data.plot()
plt.title('Forecasted Rainfall')
plt.xlabel('Date')
plt.ylabel('Rainfall in mm')
plt.show()