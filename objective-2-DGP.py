# packages
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# DGP code
# note that although this is code to generate the DGP for objective 2, the functions
# retain a previous naming convention in which this objective is called 1_b

def dgp1_b(n, L_threshold = -1.5, seed=123):
    # Set a seed for reproducibility
    np.random.seed(seed)

    # Generate the data
    M = np.random.binomial(1, 0.5, n) 
    C = np.random.normal(0, 1, n)
    epsilon_L = np.random.normal(0, 0.5, n)
    L = 0.7 * M + 0.3 * C + epsilon_L
    
    #estimand x
    x = 0.5 * C - 1.5 * L + 0.5
    
    
    Y_p = [0.1 if i < 0 else 0.9 for i in x] 
    #Y_p = sigma(x)
    
    Y = np.random.binomial(1, Y_p, n)
    
    # Combine the data into a DataFrame
    data = pd.DataFrame({'M': M, 'C': C, 'L': L, 'x': x, 'Y_p': Y_p, 'Y': Y})
    
    # Remove rows where M == 0 by L threshold
    remove_indices = data[(data['M'] == 0) & (data['L'] < L_threshold)].index #intersection of L < threshold and M == 0
    
    data = data.drop(remove_indices)
    
    # Return the generated DataFrame
    return data


#generate data, fit model, and evaluate metrics for DGP 2
def fit_and_evaluate_b(n, L_threshold):

    # 70% training, 30% testing
    data_train = dgp1_b(np.floor(0.7 * n).astype(int), L_threshold) #Generate a training set missingness by L, M
    data_test = dgp1_b(np.floor(0.3 * n).astype(int), L_threshold = -10) #Generate a training set w no missingness by L
    
    # Save training datasets as .csv files
    file_name = f"training_data_ratio_{L_threshold}.csv"
    data_train.to_csv(file_name, index=False)


    #How many M = 0 observations in training set
    M0_num = len(data_train[data_train['M'] == 0])
    # How many total observations present in training set
    num = len(data_train)
    
    # Split data into features and target
    X_train = data_train[['C', 'L', 'M']] # Using only C and L for model training
    X_test = data_test[['C', 'L', 'M']]
    y_train = data_train[['Y']]
    y_test = data_test[['Y']]
    
    # Fit logistic regression model on the training data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the testing data
    # Calculate overall accuracy and AUC
    predictions = model.predict(X_test[['C', 'L', 'M']])
    probabilities = model.predict_proba(X_test[['C', 'L', 'M']])[:, 1]
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
    predictions_M0 = model.predict(data_test_M0[['C', 'L', 'M']])
    probabilities_M0 = model.predict_proba(data_test_M0[['C', 'L', 'M']])[:, 1]
    accuracy_M0 = accuracy_score(data_test_M0['Y'], predictions_M0)
    auc_M0 = roc_auc_score(data_test_M0['Y'], probabilities_M0)

    # Calculate metrics for M == 1 group
    predictions_M1 = model.predict(data_test_M1[['C', 'L', 'M']])
    probabilities_M1 = model.predict_proba(data_test_M1[['C', 'L', 'M']])[:, 1]
    accuracy_M1 = accuracy_score(data_test_M1['Y'], predictions_M1)
    auc_M1 = roc_auc_score(data_test_M1['Y'], probabilities_M1)

    # Return all metrics
    return {
        'Training set n' : num,
        'Num M0 present in training set': M0_num,
        'Overall Accuracy': overall_accuracy,
        'Overall AUC': overall_auc,
        'Accuracy M == 0': accuracy_M0,
        'AUC M == 0': auc_M0,
        'Accuracy M == 1': accuracy_M1,
        'AUC M == 1': auc_M1
    }

#set sample size
n = 10000

#generate synthetic data and print performance metrics
# Define set of cutoff thresholds for L, below which data is missing for M==0
L_t = [np.round(x * .1, decimals = 3) for x in range(-15, 14, 4)]

# Loop over different values for threshold
for threshold in L_t:

    # Generate data, fit model and calculate model metrics
    model_metrics = fit_and_evaluate_b(n, threshold)
    
    print(f"Metrics for L cut-off threshold: L = {threshold}:")
    for metric, value in model_metrics.items():
        print(f"  {metric}: {value}")
    print()  # Print a blank line for better readability between different percentages
