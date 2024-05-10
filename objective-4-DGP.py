# packages
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import iqr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


# DGP for objective 4, strength of omitted covariate


# File paths
train_base_path = 'objective_4_data/'

test_base_path = 'objective_4_data/'


def dgp4(n, alpha):
    """ Data Generating Process for Objective 4 """
    #if alpha not in [0, 0.5, 1, 2, 4]:
     #   raise ValueError("alpha must be one of the following values: 0, 0.5, 1, 2, 4")
    
    # Generate data
    M = np.random.binomial(1, 0.5, n)
    C = np.random.normal(0, 1, n)
    epsilon_L = np.random.normal(0, 0.5, n)
    L = 0.3 * M + 0.3 * C + epsilon_L
    
    # if alpha != 0 then variation around P(Y) ~ L is proportional to C and some random noise
    #if (alpha!= 0):
     #   x = alpha * C + L - 0.2 #argument in step function
    
    # if alpha = 0 then variation around P(Y) ~ L is random noise
    #else:
     #   x = np.random.normal(0, 1, n) + L - 0.2 #argument in step function
    
    x = alpha * C + L - 0.2
    # Calculate Y_p using the logistic function with the given alpha
    Y_p = [0.1 if i < 0 else 0.9 for i in x]

    Y = np.random.binomial(1, Y_p, n)
    
    # Combine data into a DataFrame
    #data = pd.DataFrame({'M': M, 'C': C, 'L': L, 'Y_p': Y_p, 'Y': Y})
    data = pd.DataFrame({'M': M, 'L': L, 'C': C, 'x': x, 'Y_p': Y_p, 'Y': Y})
    
    return data


def fit_and_evaluate(data, alpha):
    # Split data into features and target
    X = data[['M', 'L']]  # Using only M and L for model training
    y = data['Y']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training, 30% testing

    
    # save datasets as csv files
    train_data_path = train_base_path + f'/train_alpha_{alpha}.csv'
    test_data_path = test_base_path + f'/test_alpha_{alpha}.csv'

	# re-formatting code to export .csv files to match objective-3-run.py:
    # 1. create dataset for export purposes
    X_train_export = X_train
    X_train_export['Y'] = y_train

    X_test_export = X_test
    X_test_export['Y'] = y_test

    # 2. Save training datasets as .csv files
    X_train_export.to_csv(train_data_path, index=False)
    X_test_export.to_csv(test_data_path, index=False)

    # Fit logistic regression model on the training data
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predictions on the test data
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Overall metrics on the test data
    overall_accuracy = accuracy_score(y_test, predictions)
    overall_auc = roc_auc_score(y_test, probabilities)

    # Metrics for M = 0 on the test data
    accuracy_m0 = accuracy_score(y_test[X_test['M'] == 0], predictions[X_test['M'] == 0])
    auc_m0 = roc_auc_score(y_test[X_test['M'] == 0], probabilities[X_test['M'] == 0])

    # Metrics for M = 1 on the test data
    accuracy_m1 = accuracy_score(y_test[X_test['M'] == 1], predictions[X_test['M'] == 1])
    auc_m1 = roc_auc_score(y_test[X_test['M'] == 1], probabilities[X_test['M'] == 1])

    return overall_accuracy, accuracy_m0, accuracy_m1, overall_auc, auc_m0, auc_m1



# Iterate over different alpha values
alpha_values = np.arange(0, 4.1, 0.5)
results = {}

for alpha in alpha_values:
    data = dgp4(10000, alpha)
    results[alpha] = fit_and_evaluate(data, alpha)

# Print results
for alpha, metrics in results.items():
    print(f"Alpha: {alpha}")
    print(f"Overall Accuracy: {metrics[0]}, Accuracy (M=0): {metrics[1]}, Accuracy (M=1): {metrics[2]}")
    print(f"Overall AUC: {metrics[3]}, AUC (M=0): {metrics[4]}, AUC (M=1): {metrics[5]}\n")