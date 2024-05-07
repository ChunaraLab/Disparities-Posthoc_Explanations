# packages
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# DGP code
# note: previous versions of this code refered to DGP 1 as DGP A
#note: Mratio gives the ratio (as a fraction) of M0 to M1 observations

def dgp1_a(n, Mratio = 0.5, seed=123):
    # Set a seed for reproducibility
    np.random.seed(seed)
    
    #number of M0 and M1 observations
    M0 = [0] * np.floor(Mratio * n).astype(int)
    M1 = [1] * np.floor((1-Mratio) * n).astype(int)
    
    # Generate the data
    M = np.array(M0 + M1) #concatenate both lists, convert to np.ndarray
    n_M = len(M) #since we are using np.floor above, make sure we have the same len for C, L
    C = np.random.normal(0, 1, n_M)
    epsilon_L = np.random.normal(0, 0.5, n_M)
    L = 0.7 * M + 0.3 * C + epsilon_L
    
    #estimand x
    x = 0.5 * C - 1.5 * L + 0.5
    
    
    Y_p = [0.1 if i < 0 else 0.9 for i in x] 
    #Y_p = sigma(x)
    
    Y = np.random.binomial(1, Y_p, n_M)
    
    # Combine the data into a DataFrame
    data = pd.DataFrame({'M': M, 'C': C, 'L': L, 'x': x, 'Y_p': Y_p, 'Y': Y})
    
    # Return the generated DataFrame
    return data


# code for computing model performance metrics (i.e. prediction accuracy)
def calculate_model_metrics(data, model):
    # Calculate overall accuracy and AUC
    predictions = model.predict(data[['C', 'L']])
    probabilities = model.predict_proba(data[['C', 'L']])[:, 1]
    overall_accuracy = accuracy_score(data['Y'], predictions)
    overall_auc = roc_auc_score(data['Y'], probabilities)

    # Separate the data based on the value of M
    data_M0 = data[data['M'] == 0]
    data_M1 = data[data['M'] == 1]

    # Calculate metrics for M == 0 group
    predictions_M0 = model.predict(data_M0[['C', 'L']])
    probabilities_M0 = model.predict_proba(data_M0[['C', 'L']])[:, 1]
    accuracy_M0 = accuracy_score(data_M0['Y'], predictions_M0)
    auc_M0 = roc_auc_score(data_M0['Y'], probabilities_M0)

    # Calculate metrics for M == 1 group
    predictions_M1 = model.predict(data_M1[['C', 'L']])
    probabilities_M1 = model.predict_proba(data_M1[['C', 'L']])[:, 1]
    accuracy_M1 = accuracy_score(data_M1['Y'], predictions_M1)
    auc_M1 = roc_auc_score(data_M1['Y'], probabilities_M1)

    # Return all metrics
    return {
        'Overall Accuracy': overall_accuracy,
        'Overall AUC': overall_auc,
        'Accuracy M == 0': accuracy_M0,
        'AUC M == 0': auc_M0,
        'Accuracy M == 1': accuracy_M1,
        'AUC M == 1': auc_M1
    }





#generate data, fit model, and evaluate prediction metrics for DGP 1
def fit_and_evaluate_a(n, Mratio):

    # 70% training, 30% testing
    data_train = dgp1_a(np.floor(0.7 * n), Mratio) #Generate a training set with a skewed ratio of M0 to M1
    data_test = dgp1_a(np.floor(0.3 * n), Mratio = 0.5) #Generate a testing set with 50/50 M0 and M1

    # Save training datasets as .csv files
    file_name = f"training_data_ratio_{Mratio}.csv"
    data_train.to_csv(file_name, index=False)

    
    # Split data into features and target
    X_train = data_train[['C', 'L']] # Using only C and L for model training
    X_test = data_test[['C', 'L']]
    y_train = data_train[['Y']]
    y_test = data_test[['Y']]
    
    # Fit logistic regression model on the training data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the testing data
    # Calculate overall accuracy and AUC
    predictions = model.predict(X_test[['C', 'L']])
    probabilities = model.predict_proba(X_test[['C', 'L']])[:, 1]
    overall_accuracy = accuracy_score(y_test['Y'], predictions)
    overall_auc = roc_auc_score(y_test['Y'], probabilities)

    # Separate the data based on the value of M
    data_test_M0 = data_test[data_test['M'] == 0]
    data_test_M1 = data_test[data_test['M'] == 1]
    #X_test_M0 = data_test_M0[['C', 'L']]
    #X_test_M1 = data_test_M1[['C', 'L']]
    #y_test_M0 = data_test_M0[['Y']]
    #y_test_M1 = data_test_M1[['Y']]

    # Calculate metrics for M == 0 group
    predictions_M0 = model.predict(data_test_M0[['C', 'L']])
    probabilities_M0 = model.predict_proba(data_test_M0[['C', 'L']])[:, 1]
    accuracy_M0 = accuracy_score(data_test_M0['Y'], predictions_M0)
    auc_M0 = roc_auc_score(data_test_M0['Y'], probabilities_M0)

    # Calculate metrics for M == 1 group
    predictions_M1 = model.predict(data_test_M1[['C', 'L']])
    probabilities_M1 = model.predict_proba(data_test_M1[['C', 'L']])[:, 1]
    accuracy_M1 = accuracy_score(data_test_M1['Y'], predictions_M1)
    auc_M1 = roc_auc_score(data_test_M1['Y'], probabilities_M1)

    # Return all metrics
    return {
        'Overall Accuracy': overall_accuracy,
        'Overall AUC': overall_auc,
        'Accuracy M == 0': accuracy_M0,
        'AUC M == 0': auc_M0,
        'Accuracy M == 1': accuracy_M1,
        'AUC M == 1': auc_M1
    }


# Set the sample size, create synthetic data and save, print performance metrics
n = 10000
# Define set of ratios to concider for M0:M1
ratios = [np.round(x * .05, decimals = 3) for x in range(10, 0, -1)]

# Loop over different values for ratio
for ratio in ratios:

    # Generate data, fit model and calculate model metrics
    model_metrics = fit_and_evaluate_a(n, ratio)
    
    print(f"Metrics for ratio = {ratio}%:")
    for metric, value in model_metrics.items():
        print(f"  {metric}: {value}")
    print()  # Print a blank line for better readability between different percentages

