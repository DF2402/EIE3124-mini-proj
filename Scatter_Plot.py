import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Read the dataset
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv('boston.csv', skiprows=1, names=names)

# Identify missing values
missing_values = dataset.isnull().sum()

# Remove missing values
dataset = dataset.dropna()

# Identify outliers
plt.figure(figsize=(10, 6))
dataset.boxplot(column=names)
plt.xticks(rotation=45)
plt.show()

# Remove outliers
z_scores = np.abs((dataset - dataset.mean()) / dataset.std())
dataset = dataset[(z_scores < 3).all(axis=1)]

# Impute missing values (Replace missing values with mean)
dataset = dataset.fillna(dataset.mean())

dataset.to_csv('output.csv', index=False)

# Scatter plot and fitted curve for LSTAT
plt.figure(figsize=(10, 6))
plt.scatter(dataset['LSTAT'], dataset['MEDV'], c='b', label='LSTAT')

# Fit a polynomial curve
coefficients = np.polyfit(dataset['LSTAT'], dataset['MEDV'], 2)
x_values = np.linspace(dataset['LSTAT'].min(), dataset['LSTAT'].max(), 100)
y_values = np.polyval(coefficients, x_values)
plt.plot(x_values, y_values, c='r', label='Fitted Curve')

plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.legend()
plt.show()

# Scatter plot and fitted curves for RM
plt.figure(figsize=(10, 6))
plt.scatter(dataset['RM'], dataset['MEDV'], c='r', label='RM')

slope, intercept, r_value, p_value, std_err = linregress(dataset['RM'], dataset['MEDV'])
x_values = np.linspace(dataset['RM'].min(), dataset['RM'].max(), 100)
y_values = slope * x_values + intercept
plt.plot(x_values, y_values, c='b', label='Fitted Curve')

plt.xlabel('RM')
plt.ylabel('MEDV')
plt.legend()
plt.show()

# Scatter plot and fitted curves for NOX
plt.figure(figsize=(10, 6))
plt.scatter(dataset['NOX'], dataset['MEDV'], c='g', label='NOX')

slope, intercept, r_value, p_value, std_err = linregress(dataset['NOX'], dataset['MEDV'])
x_values = np.linspace(dataset['NOX'].min(), dataset['NOX'].max(), 100)
y_values = slope * x_values + intercept
plt.plot(x_values, y_values, c='b', label='Fitted Curve')

plt.xlabel('NOX')
plt.ylabel('MEDV')
plt.legend()
plt.show()

# Scatter plot and fitted curves for TAX
plt.figure(figsize=(10, 6))
plt.scatter(dataset['TAX'], dataset['MEDV'], c='m', label='TAX')

slope, intercept, r_value, p_value, std_err = linregress(dataset['TAX'], dataset['MEDV'])
x_values = np.linspace(dataset['TAX'].min(), dataset['TAX'].max(), 100)
y_values = slope * x_values + intercept
plt.plot(x_values, y_values, c='b', label='Fitted Curve')

plt.xlabel('TAX')
plt.ylabel('MEDV')
plt.legend()
plt.show()

# Pearson correlation coefficient
correlation_matrix = dataset[['LSTAT', 'RM', 'NOX', 'TAX', 'MEDV']].corr()
pearson_coefficient = correlation_matrix.loc['LSTAT', 'MEDV']
print("Pearson correlation coefficient between LSTAT and MEDV:", pearson_coefficient)

pearson_coefficient = correlation_matrix.loc['RM', 'MEDV']
print("Pearson correlation coefficient between RM and MEDV:", pearson_coefficient)

pearson_coefficient = correlation_matrix.loc['NOX', 'MEDV']
print("Pearson correlation coefficient between NOX and MEDV:", pearson_coefficient)

pearson_coefficient = correlation_matrix.loc['TAX', 'MEDV']
print("Pearson correlation coefficient between TAX and MEDV:", pearson_coefficient)
