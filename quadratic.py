import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

names = ['LSTAT','RM','MEDV']
dataset = pd.read_csv('boston.csv', usecols=names)


# division labels
X = dataset[['LSTAT', 'RM']].values
y = dataset['MEDV'].values
# divide training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

#linear regression
model = LinearRegression()
# fit quadratic regression
model.fit(X_train_poly, y_train)
# Combine the coefficients and intercept into a single array
weights = np.append(model.coef_, model.intercept_)
# Print the combined weights
print('Weights:', ['{:.4f}'.format(w) for w in weights])

#calculate prediction results
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

#regression function
# Construct the polynomial equation as a string
equation_terms = []
feature_names = poly.get_feature_names_out(['LSTAT', 'RM'])
for coef, feature_name in zip(weights[:-1], feature_names):
    equation_terms.append(f"{coef:.2f}*{feature_name}")
equation = " + ".join(equation_terms)
equation = f"MEDV = {equation} + {weights[-1]:.2f}"
print(equation)
#error calculation
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"MSE for training:{mse_train}")
print(f"MSE for testing:{mse_test}")


#plot the fitting line in 3d plot
# Plotting the 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a grid over the feature space
x_surf = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_surf = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)

# Transform the grid using the same polynomial features
poly_surf = poly.transform(np.c_[x_surf.ravel(), y_surf.ravel()])
z_surf = model.predict(poly_surf).reshape(x_surf.shape)

# Plot the surface
ax.plot_surface(x_surf, y_surf, z_surf, cmap='viridis', alpha=0.6)

# Scatter plot of training data
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color='red', label='Training data')

# Scatter plot of testing data
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='blue', label='Testing data', alpha=0.5)

# Labels and title
ax.set_xlabel(names[0])
ax.set_ylabel(names[1])
ax.set_zlabel('MEDV')
ax.set_title(f'Polynomial Fit Equation:\n\n{equation}',fontsize=8)
# Legend
ax.legend()

# Show plot
plt.show()
plt.close()

#2d scatter plot
#for testing
plt.plot(y_test_pred,'bo',label = 'predict')
plt.plot(y_test,'ro', label = 'actual value')
plt.ylabel('MEDV')
plt.title('MEDV regression for testing dataset')
plt.legend()
plt.show()
plt.close()

plt.plot(y_train_pred,'bo',label = 'predict')
plt.plot(y_train,'ro', label = 'actual value')
plt.ylabel('MEDV')
plt.title('MEDV regression for training dataset')
plt.legend()
plt.show()
plt.close()


