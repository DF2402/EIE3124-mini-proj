import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import linregress
from scipy import stats

# Read the cleaned dataset
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv('cleaned_dataset.csv')

# division labels
X = dataset[['LSTAT', 'RM']].values
y = dataset['MEDV'].values
# divide training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

#linear regression
model = LinearRegression()
# train linear regression model
model.fit(X_train, y_train)
w1 = np.insert(model.coef_, 0, model.intercept_)
# Print the combined weights
print('Weights for linear regression:',[f"{w:.8f}" for w in w1])

#conduct MAP
class LinearRegressionMAP:
    def __init__(self):
        self.weights = None

    def train(self, train_x, train_y, lambd):
        bias = np.ones((train_x.shape[0], 1))
        X = np.concatenate((train_x, bias), axis=1)
        self.weights = np.linalg.inv(X.T.dot(X) + lambd * np.eye(X.shape[1])).dot(X.T).dot(train_y)

    def predict(self, test_x):
        bias = np.ones((test_x.shape[0], 1))
        X = np.concatenate((test_x, bias), axis=1)
        return X.dot(self.weights)
    def get_weights(self):
        return self.weights
map_model = LinearRegressionMAP()
lambd = 0.1
map_model.train(X_train, y_train, lambd)
w2=map_model.get_weights()
w2_ordered = np.insert(w2[:-1], 0, w2[-1])
print('Weights for MAP Method:',w2_ordered)

# map prediction result for test dataset
y_pmap_test = map_model.predict(X_test)
# linear regression result for test dataset
y_plr_test = model.predict(X_test)
# map prediction result for test dataset
y_pmap_train = map_model.predict(X_train)
# linear regression result for test dataset
y_plr_train = model.predict(X_train)

# Create a 3D subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for actual data
ax.scatter(dataset['RM'], dataset['LSTAT'], dataset['MEDV'], color="blue", label='Actual MEDV')

# We need to create a mesh of x values (RM and LSTAT) to plot the surface.
rm = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), len(X))
lstat = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), len(X))
rm, lstat = np.meshgrid(rm, lstat)

# Now we use our model to predict the values across the grid
z = map_model.predict(np.c_[lstat.ravel(), rm.ravel()])
z = z.reshape(rm.shape)

# Plot the surface for predicted data
ax.plot_surface(rm, lstat, z, color="red", alpha=0.7, label="Predicted MEDV")

# Set labels and title
ax.set_xlabel('RM')
ax.set_ylabel('LSTAT')
ax.set_zlabel('MEDV')
ax.set_title('Actual vs Predicted MEDV')
plt.show()
plt.close()

# Show plot for MAP on test sets
plt.plot(y_pmap_train,'bo',label = 'predict')
plt.plot(y_train,'ro', label = 'actual value')
plt.ylabel('MEDV')
plt.legend()
plt.title("MAP on train dataset")
plt.show()
plt.close()
# Show plot for MAP on test sets
plt.plot(y_pmap_test,'bo',label = 'predict')
plt.plot(y_test,'ro', label = 'actual value')
plt.ylabel('MEDV')
plt.legend()
plt.title("MAP on test dataset")
plt.show()
plt.close()

# evaluation metrics
mse_map_test = mean_squared_error(y_pmap_test,y_test)
mse_map_train= mean_squared_error( y_pmap_train,y_train)
mse_lr_test = mean_squared_error(y_plr_test,y_test)
mse_lr_train = mean_squared_error(y_plr_train,y_train)

lse_map = calculate_lse(y_test,y_pmap_test)
print("LSE (MAP):", lse_map)
r2_map = calculate_r2(y_test,y_pmap_test)
print("R2 (MAP):", r2_map)

lse_lr = calculate_lse(y_test,y_plr_train)
print("LSE (Linear regression):", lse_lr)
r2_lr = calculate_r2(y_test,y_plr_train)
print("R2 (Linear regression):", r2_lr)

print(f"MSE for training on MAP model:{mse_map_train}")
print(f"MSE for testing on MAP model:{mse_map_test}")
print(f"MSE for training on linear regression model:{mse_lr_train}")
print(f"MSE for testing on linear regression model:{mse_lr_test}")

"""
# RMSE
rmse1 = mean_squared_error(y_test, y_pred1, squared=False)
rmse2 = mean_squared_error(y_test, y_pred2, squared=False)
print(f"RMSE: {rmse2}")

# MAE
mae = mean_absolute_error(y_test, y_pred1)
print(f"Mean Absolute Error (MAE): {mae}")

# R²
r1 = r2_score(y_test, y_pred1)
r2 = r2_score(y_test, y_pred2)
print()
"""


