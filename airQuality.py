
# Objective : Air Quality Index Prediction
# By : Mohit Upadhyay

import pandas as pd
import numpy as np


# Load the dataset
data = pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\EIProject\\code\\airQualityIndia.csv')

# Display the first few rows of the datasetg
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing target values (AQI)
data.dropna(subset=['AQI'], inplace=True)



# data analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of AQI
plt.figure(figsize=(10, 6))
sns.histplot(data['AQI'], kde=True)
plt.title('Distribution of AQI')
plt.xlabel('AQI')
plt.ylabel('Frequency')
plt.show()

# conversion of date 
data['Date'] = pd.to_datetime(data['Date'])
data_numeric = data.select_dtypes(include=[np.number])



# Plot correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = data_numeric.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()



# Select features for prediction
features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
             'Benzene', 'Toluene', 'Xylene']
target = 'AQI'

X = data[features]
y = data[target]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("*******")
print("X : ",X)
print("y : ",y)
# data modelling

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize and train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print("---------------")
# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')



# data visualization

# Plot actual vs predicted AQI
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted AQI')
plt.show()
