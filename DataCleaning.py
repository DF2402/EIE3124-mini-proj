import pandas as pd
from scipy.stats.mstats import winsorize

# Read the dataset
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv('boston.csv', skiprows=1, names=names)

# Identify missing valuesa
missing_values = dataset.isnull().sum()

# Remove missing values
dataset = dataset.dropna()
# Impute missing values (Replace missing values with mean)
dataset = dataset.fillna(dataset.mean())

# Winsorize:upper 10% and lower 2% are replaced
winsorized_MEDV = winsorize(dataset['MEDV'], limits=[0.02, 0.1])

# put the winsorized data back to datasets
dataset['MEDV'] = winsorized_MEDV
name = ['LSTAT','RM','MEDV']
dataset = dataset[name]
dataset.to_csv('cleaned_dataset.csv', index=False)
print("The dataset is clean. ")