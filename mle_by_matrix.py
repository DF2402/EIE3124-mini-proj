# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 22:43:31 2024

@author: Administrator
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from evaluation import calculate_lse,calculate_r2
import matplotlib.pyplot as plt




# Reading features from the dataset
names = ['LSTAT', 'RM', 'MEDV']
dataset = pd.read_csv('boston.csv', usecols=names)

# Splitting features and labels
X = dataset[['LSTAT', 'RM']].values
y = dataset['MEDV'].values

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)
X_train_df = pd.DataFrame(X_train,columns=['LSTAT', 'RM'])
Y_train_df = pd.DataFrame(y_train,columns=['MEDV'])
training_data = pd.concat([X_train_df,Y_train_df],axis = 1)

X_test_df = pd.DataFrame(X_test,columns=['LSTAT', 'RM'])
Y_test_df = pd.DataFrame(y_test,columns=['MEDV'])
testing_data = pd.concat([X_test_df,Y_test_df],axis = 1)

def mle_by_matrix(dataset):
    y_train = dataset['MEDV']
    dataset = dataset.drop('MEDV', axis=1)
    # Adding an intercept term to the training data
    X_with_intercept = np.column_stack((np.ones(dataset.shape[0]), dataset))
    
    # Computing the estimated values of w
    XTX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    w = XTX_inv @ X_with_intercept.T @ y_train
    
    # Outputting w
    return w

def k_fold (dataset,k):
    print (f"K-fold Cross-Validation K = {k}")
    # K-fold Cross-Validation
    r_squares = []
    LSEs = []
    models = []
    #Splitting subset
    subsets = np.array_split(dataset, k)
    
    for subset in subsets:
        
        # distribute training and testing
        testing_set = subset
        
        training_set = dataset.drop(testing_set.index)
        model = mle_by_matrix(training_set)
        predict = []
        n = training_set.shape[0]
        for index, row in testing_set.iterrows():
            temp_x,temp_z,temp_y = row['LSTAT'],row['RM'],row['MEDV']
            predict.append(regression(temp_x,temp_z,model))
        # model performance
        r_square = calculate_r2(testing_set['MEDV'],predict)
        r_squares.append(r_square)
        LSE = calculate_lse(testing_set['MEDV'],predict)
        LSEs.append(LSE)
        models.append(model)
        
    print("coefficients of models :", models)
    print("r^2 :",r_squares)
    print("LSE :",LSEs)
    
    best_r_squares = max(r_squares)
    index_of_best_model = r_squares.index(best_r_squares)
    
    return models[index_of_best_model]

def regression (x,z,model):
    #training with parameters
    w0,w1,w2 = model 
    return w0 + w1 * x + w2*z


model_k = k_fold(training_data,5)
# testing 
predict = []
    
n = testing_data.shape[0]
for i in range(n) :
    temp_x = testing_data['LSTAT'][i]
    temp_z = testing_data['RM'][i]
    predict.append(regression(temp_x,temp_z,model_k))
# plot
plt.plot(predict,'bo',label = 'predict')
plt.plot(testing_data['MEDV'],'ro', label = 'actual value')
plt.ylabel('MEDV')
plt.legend()
plt.title("K-fold")
plt.show()

# evaluation
r_square = calculate_r2(testing_data['MEDV'],predict)
LSE = calculate_lse(testing_data['MEDV'],predict)
print('k-fold r^2 : ',r_square," LSE : ",LSE)

model = mle_by_matrix(training_data)
print(model)

# testing 
predict = []
    
n = testing_data.shape[0]
for i in range(n) :
    temp_x = testing_data['LSTAT'][i]
    temp_z = testing_data['RM'][i]
    predict.append(regression(temp_x,temp_z,model))
     
# plot
plt.plot(predict,'bo',label = 'predict')
plt.plot(testing_data['MEDV'],'ro', label = 'actual value')
plt.ylabel('MEDV')
plt.legend()
plt.show()

# evaluation
r_square = calculate_r2(testing_data['MEDV'],predict)
LSE = calculate_lse(testing_data['MEDV'],predict)
print('normal r^2 : ',r_square," LSE : ", LSE)
