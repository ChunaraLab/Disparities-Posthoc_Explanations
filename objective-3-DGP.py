# packages
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


# DGP for concept shift (objective 3)
def dgp3(n, beta):
    """Data Generating Process for Objective 2."""
    M = np.random.binomial(1, 0.5, n)
    C = np.random.normal(0, 1, n)
    epsilon_L = np.random.normal(0, 0.1, n)
    L = 0.7 * M + 0.3 * C + epsilon_L

    #estimand x
    #x = 0.5 * C + 1.5 * M * L + beta * (1-M) * L - 1 * M
    x = 0.5 * C + 1.5 * M * L + beta * (1-M) * L - 1 * L - 0.2
    
    Y_p = [0.1 if i < 0 else 0.9 for i in x]
      
    Y = np.random.binomial(1, Y_p, n)
    
    data = pd.DataFrame({
        'M': M,
        'C': C,
        'L': L,
        'x': x,
        'Y_p': Y_p,
        'Y': Y
    })

    return data

BASE_PATH = '/objective_3_data/'

# case A, black box model includes all features Ml, L , C and interaction between L and M
def fit_and_evaluate_a(data, beta):
    #create interaction term
    data['L*M'] = data['L'] * data['M']
    
    # Split data into features and target
    X = data[['C', 'L', 'M', 'L*M']]  # Model has access to all components of DGP
    y = data['Y']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training, 30% testing

    # re-formatting code to export .csv files to match objective-3-run.py:
    # 1. create dataset for export purposes
    X_train_export = X_train
    X_train_export['Y'] = y_train
    
    X_test_export = X_test
    X_test_export['Y'] = y_test

    # 2. Save datasets as .csv files
    file_name = f"training_data_{beta}.csv"
    train_data_path = BASE_PATH + file_name
    data_train.to_csv(train_data_path, index=False)
    
    file_name = f"testing_data_{beta}.csv"
    test_data_path = BASE_PATH + file_name
    X_test_export.to_csv(test_data_path, index=False)

    
    # Fit logistic regression model on the training data
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict on the testing data
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Calculate overall accuracy and AUC
    overall_accuracy = accuracy_score(y_test, y_pred)
    overall_auc = roc_auc_score(y_test, y_pred_prob)

    # Calculate accuracy and AUC for data where M == 0
    accuracy_m0 = accuracy_score(y_test[data.loc[y_test.index, 'M'] == 0], y_pred[data.loc[y_test.index, 'M'] == 0])
    auc_m0 = roc_auc_score(y_test[data.loc[y_test.index, 'M'] == 0], y_pred_prob[data.loc[y_test.index, 'M'] == 0])

    # Calculate accuracy and AUC for data where M == 1
    accuracy_m1 = accuracy_score(y_test[data.loc[y_test.index, 'M'] == 1], y_pred[data.loc[y_test.index, 'M'] == 1])
    auc_m1 = roc_auc_score(y_test[data.loc[y_test.index, 'M'] == 1], y_pred_prob[data.loc[y_test.index, 'M'] == 1])

    
    # Return all metrics
    return {
        'Overall Accuracy': overall_accuracy,
        'Overall AUC': overall_auc,
        'Accuracy M == 0': accuracy_m0,
        'AUC M == 0': auc_m0,
        'Accuracy M == 1': accuracy_m1,
        'AUC M == 1': auc_m1
    }




# run code:

# set values for betas
betas = [-2.7, -0.7, -0.5, 0.5, 1.5, 2.7]



# Loop over different values beta
for beta in betas:
    # Generate data
    my_data_objective3 = dgp3(20000, beta)

    # Fit model and calculate model metrics
    model_metrics = fit_and_evaluate_a(my_data_objective3, beta)
    
    print(f"Metrics for beta = {beta}%:")
    for metric, value in model_metrics.items():
        print(f"  {metric}: {value}")
    print()  # Print a blank line for better readability between different percentages

############################################################################


# case b, no subgroup information (M) in black box models
# note: these datasets are generated to evaluate model prediction performance
# and this code is not needed to generate datasets for objective-3-run.py


def fit_and_evaluate_b(data, beta):
    
    # Split data into features and target
    X = data[['C', 'L']]  # Model only includes L and C
    y = data['Y']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training, 30% testing
    
    # Fit logistic regression model on the training data
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict on the testing data
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Calculate overall accuracy and AUC
    overall_accuracy = accuracy_score(y_test, y_pred)
    overall_auc = roc_auc_score(y_test, y_pred_prob)

    # Calculate accuracy and AUC for data where M == 0
    accuracy_m0 = accuracy_score(y_test[data.loc[y_test.index, 'M'] == 0], y_pred[data.loc[y_test.index, 'M'] == 0])
    auc_m0 = roc_auc_score(y_test[data.loc[y_test.index, 'M'] == 0], y_pred_prob[data.loc[y_test.index, 'M'] == 0])

    # Calculate accuracy and AUC for data where M == 1
    accuracy_m1 = accuracy_score(y_test[data.loc[y_test.index, 'M'] == 1], y_pred[data.loc[y_test.index, 'M'] == 1])
    auc_m1 = roc_auc_score(y_test[data.loc[y_test.index, 'M'] == 1], y_pred_prob[data.loc[y_test.index, 'M'] == 1])

    
    # Return all metrics
    return {
        'Overall Accuracy': overall_accuracy,
        'Overall AUC': overall_auc,
        'Accuracy M == 0': accuracy_m0,
        'AUC M == 0': auc_m0,
        'Accuracy M == 1': accuracy_m1,
        'AUC M == 1': auc_m1
    }
    
betas = [-2.7, -0.7, -0.5, 0.5, 1.5, 2.7]



# Loop over different values beta
for beta in betas:
    # Generate data
    my_data_objective3 = dgp3(10000, beta)

    # Fit model and calculate model metrics
    model_metrics = fit_and_evaluate_b(my_data_objective3, beta)
    
    print(f"Metrics for beta = {beta}%:")
    for metric, value in model_metrics.items():
        print(f"  {metric}: {value}")
    print()  # Print a blank line for better readability between different percentages