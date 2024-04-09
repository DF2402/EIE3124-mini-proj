import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Function to calculate Least Squared Error (LSE)
def calculate_lse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

# Function to calculate R2 (R-squared)
def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

if __name__ == "__main__":
    # Calculate LSE for MLE model
    y_pred_mle = np.array([regression(x[0], x[1], model) for x in X_test])
    lse_mle = calculate_lse(y_test, y_pred_mle)
    print("LSE (MLE):", lse_mle)
    
    # Calculate LSE for MAP model
    y_pred_map = np.array([regression(x[0], x[1], model_k) for x in X_test])
    lse_map = calculate_lse(y_test, y_pred_map)
    print("LSE (MAP):", lse_map)
    
    
    
    # Calculate R2 for MLE model
    r2_mle = calculate_r2(y_test, y_pred_mle)
    print("R2 (MLE):", r2_mle)
    
    # Calculate R2 for MAP model
    r2_map = calculate_r2(y_test, y_pred_map)
    print("R2 (MAP):", r2_map)
