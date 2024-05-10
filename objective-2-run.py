import os
import numpy as np
import argparse
import scipy.stats as stats
import matplotlib.pyplot as plt
from data_preparation import DataProcessor
from model import LR, NN
from explanation import LimeExplanations
from evaluation_metrics import EvaluationMetrics
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

## script 
#python main.py --path "/Users/salman/OneDrive-nyu.edu (Archive)/NYU/prof_chunara/multi level model/explanation-disparity-main" --dataset "synthetic" --setup 1

def confidence_interval(data, confidence=0.95):
    """Compute the confidence interval for a given data set using the t-distribution."""
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    
    # Calculate the t-score for the given confidence level and degrees of freedom
    confidence_level = 0.95

    ci = stats.t.interval(confidence_level, len(data)-1, loc=mean, scale=sem)
    
    return (ci)

def plot_accuracies(results, save_path):
    removal_rates = list(results.keys())
    overall_accuracies = [results[rate]['mean_overall_accuracy'] for rate in removal_rates]
    accuracy_1 = [results[rate]['mean_accuracy_1'] for rate in removal_rates]
    accuracy_0 = [results[rate]['mean_accuracy_0'] for rate in removal_rates]

    plt.figure(figsize=(12, 8))
    plt.plot(removal_rates, overall_accuracies, label='Overall Accuracy', marker='o', markersize=8, linewidth=2)
    plt.plot(removal_rates, accuracy_1, label='Accuracy for 1', marker='x', markersize=8, linewidth=2)
    plt.plot(removal_rates, accuracy_0, label='Accuracy for 0', marker='s', markersize=8, linewidth=2)

    plt.xlabel('L threshold, s', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(removal_rates, [f"{rate}" for rate in removal_rates], fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    #plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=300)  # Save the figure
    #plt.show()



def plot_fidelity_metric_with_ci(results, title, mean_key, ci_key, save_path, bar_width=0.4):
    removal_rates = sorted(results.keys())
    mean_values = [results[rate][mean_key] for rate in removal_rates]
    ci_values = [results[rate][ci_key] for rate in removal_rates]

    lower_errors = [mean - ci[0] for mean, ci in zip(mean_values, ci_values)]
    upper_errors = [ci[1] - mean for mean, ci in zip(mean_values, ci_values)]
    errors = [lower_errors, upper_errors]

    plt.figure(figsize=(12, 8))
    plt.bar(removal_rates, mean_values, yerr=errors, capsize=5, color='skyblue', edgecolor='black')
    plt.xlabel('L threshold, s', fontsize=14)
    plt.ylabel(title, fontsize=14)
    #plt.title(f'{title} vs Percent Removal', fontsize=16)
    plt.xticks(removal_rates, [f'{rate}' for rate in removal_rates], fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=300)  # Save the figure
    #plt.show()

#objective 1a
# FILENAMES = [
#     'training_data_ratio_-0.3.csv',
#     'training_data_ratio_-0.7.csv',
#     'training_data_ratio_-1.1.csv',
#     'training_data_ratio_-1.5.csv',
#     'training_data_ratio_0.1.csv',
#     'training_data_ratio_0.5.csv',
#     'training_data_ratio_0.9.csv',
#     'training_data_ratio_1.3.csv'
# ]
    
FILENAMES = [
    'training_data_ratio_-5.csv',
    'training_data_ratio_-1.35.csv',
    'training_data_ratio_-0.4.csv',
    'training_data_ratio_0.45.csv',
    'training_data_ratio_0.9.csv',
    'training_data_ratio_1.35.csv',
    'training_data_ratio_1.75.csv'
]


# FILENAMES = [
#     'training_data_ratio_-0.3.csv',
#     'training_data_ratio_-0.7.csv'
# ]

BASE_PATH = '/objective_2_data/'

test_data_path = '/objective_2_data/testing_data_objective_2.csv'



def main(args):
    results = {}  # This dictionary will store the results for each file

    for fname in FILENAMES:
        train_data_path = BASE_PATH + fname
        #print(f"Processing file: {train_data_path}")

        # Extract the percent removal from the filename
        percent_removal = fname.split('_')[3].replace('.csv', '')  # Extracts the number after 'removal'
        print(percent_removal)

        print(f"Results for Percent Removal = {percent_removal}%")

        total_trials = 5

        # Initialize lists for metrics
        overall_accuracies = []
        group_accuracies = {}
        gt_fidelities = {}
        max_fid_gaps = []
        mean_fid_gaps_auroc = []
        mean_fid_gaps_accuracy = []
        mean_fid_gaps_residual_error = []

        for trial in range(total_trials):
            #print(f"\n--- Trial {trial+1} ---")

            # 1. Load the dataset
            seed = 123 + trial
            sensitive_attribute = args.sensitive
            processor = DataProcessor(train_data_path, test_data_path, sensitive_attribute, setup= args.setup, group_attribute_c = True)
            target = 'Y'
            X_train, X_test, y_train, y_test, indices_dict, cat_feat_indices = processor.process_data(target)

            # 2. Train the model
            if args.model == 'lr':
                model = LR(X_train.shape[1])
            elif args.model == 'nn':
                model = NN(X_train.shape[1])
            else:
                raise ValueError(f"Unsupported model: {args.model}")
            model.train(X_train.values, y_train.values)

            # 3. Evaluate the model
            accuracies = model.evaluate(X_test.values, y_test.values, indices_dict)

            lime_exp = LimeExplanations(X_train.values, cat_feat_indices, X_test.values, model.predict_proba, n_samples=1000, random_state = seed)
            overall_preds = lime_exp.get_overall_predictions()
            group_preds = lime_exp.get_group_predictions(indices_dict)
            group_proba_explanations = lime_exp.get_group_explanation_proba(indices_dict)



            eval_metrics = EvaluationMetrics(
            model = model,
            X_test = X_test.values,
            y_true = y_test, 
            indices_dict = indices_dict, 
            w = model.net[0].weight.detach().numpy().flatten(), 
            k = args.k, 
            exps = lime_exp.explanations, 
            overall_preds = overall_preds, 
            group_preds = group_preds, 
            group_explanation_preds_proba = group_proba_explanations)


            # Initialize dictionary for storing Ground Truth Fidelity for each group in each trial
            gt_fidelities = {group: [] for group in indices_dict.keys()}
            gt_fidelities['all'] = []

            # Save overall accuracy
            overall_accuracies.append(accuracies['overall'])

            # Save group accuracies
            for group in accuracies:
                if group == 'overall':
                    continue
                if group not in group_accuracies:
                    group_accuracies[group] = []
                group_accuracies[group].append(accuracies[group])

            # Calculate metrics
            max_fid_gap, _, _ = eval_metrics.maximum_fidelity_gap()
            mean_fid_gap_auroc = eval_metrics.mean_fidelity_gap_auroc()
            mean_fid_gap_accuracy = eval_metrics.mean_fidelity_gap_accuracy()
            mean_fid_gap_residual_error = eval_metrics.mean_fidelity_gap_residual_error()

            # Save metrics
            max_fid_gaps.append(max_fid_gap)
            mean_fid_gaps_auroc.append(mean_fid_gap_auroc)
            mean_fid_gaps_accuracy.append(mean_fid_gap_accuracy)
            mean_fid_gaps_residual_error.append(mean_fid_gap_residual_error)

        # Calculate mean of metrics
        mean_overall_accuracy = np.mean(overall_accuracies)
        mean_group_accuracies = {group: np.mean(acc) for group, acc in group_accuracies.items()}
        #mean_gt_fidelities = {group: np.mean(fid) for group, fid in gt_fidelities.items()}
        mean_max_fid_gap = np.mean(max_fid_gaps)
        mean_fid_gap_auroc = np.mean(mean_fid_gaps_auroc)
        mean_fid_gap_accuracy = np.mean(mean_fid_gaps_accuracy)
        mean_fid_gap_residual_error = np.mean(mean_fid_gaps_residual_error)

        # Calculate confidence intervals
        ci_max_fid_gap = confidence_interval(max_fid_gaps)
        ci_mean_fid_gap_auroc = confidence_interval(mean_fid_gaps_auroc)
        ci_mean_fid_gap_accuracy = confidence_interval(mean_fid_gaps_accuracy)
        ci_mean_fid_gap_residual_error = confidence_interval(mean_fid_gaps_residual_error)


        # Print results
        print(f'\n--- Final Results, Model: {args.model}, Setup: {args.setup}, Sensitive Attribute: {args.sensitive}, Percent Removal: {percent_removal}% ---')
        print(f"Mean Overall Accuracy: {mean_overall_accuracy}")
        for group, accuracy in mean_group_accuracies.items():
            print(f"Mean Accuracy for {group}: {accuracy}")
        #for group, fidelity in mean_gt_fidelities.items():
            #print(f"Mean Ground Truth Fidelity for {group}: {fidelity}")
        print(f"Mean Maximum Fidelity Gap: {mean_max_fid_gap}")
        print(f"Mean Fidelity Gap AUROC: {mean_fid_gap_auroc}")
        print(f"Mean Fidelity Gap Accuracy: {mean_fid_gap_accuracy}")
        print(f"Mean Fidelity Gap Residual Error: {mean_fid_gap_residual_error}")


        # Print confidence intervals
        print(f"Confidence Interval for Maximum Fidelity Gap: {ci_max_fid_gap}")
        print(f"Confidence Interval for Fidelity Gap AUROC: {ci_mean_fid_gap_auroc}")
        print(f"Confidence Interval for Fidelity Gap Accuracy: {ci_mean_fid_gap_accuracy}")
        print(f"Confidence Interval for Fidelity Gap Residual Error: {ci_mean_fid_gap_residual_error}")

        # Store metrics in the results dictionary
        results[percent_removal] = {
            'mean_overall_accuracy': mean_overall_accuracy,
            'mean_accuracy_1': mean_group_accuracies.get(1, None),
            'mean_accuracy_0': mean_group_accuracies.get(0, None),
            'mean_max_fid_gap': mean_max_fid_gap,
            'ci_max_fid_gap': ci_max_fid_gap,
            'mean_fid_gap_auroc': mean_fid_gap_auroc,
            'ci_fid_gap_auroc': ci_mean_fid_gap_auroc,
            'mean_fid_gap_accuracy': mean_fid_gap_accuracy,
            'ci_fid_gap_accuracy': ci_mean_fid_gap_accuracy,
            'mean_fid_gap_residual_error': mean_fid_gap_residual_error,
            'ci_fid_gap_residual_error': ci_mean_fid_gap_residual_error
        }

    print(results)
    # # Define the base directory path
    # base_plot_path = "/Users/salman/Desktop/exp_qua/new_3_objective/dgp_final/plot_data/objective_1_final/plot-case-b/setup-2-with-M"

    # # Define file names
    # accuracy_file = "accuracy.png"
    # max_fid_gap_file = "max_fid_gap.png"
    # fid_gap_auroc_file = "fid_gap_auroc.png"
    # fid_gap_accuracy_file = "fid_gap_accuracy.png"
    # fid_gap_residual_error_file = "fid_gap_residual_error.png"

    # # Join the base path with each file name
    # accuracy_fig_path = os.path.join(base_plot_path, accuracy_file)
    # max_fid_gap_fig_path = os.path.join(base_plot_path, max_fid_gap_file)
    # fid_gap_auroc_fig_path = os.path.join(base_plot_path, fid_gap_auroc_file)
    # fid_gap_accuracy_fig_path = os.path.join(base_plot_path, fid_gap_accuracy_file)
    # fid_gap_residual_error_fig_path = os.path.join(base_plot_path, fid_gap_residual_error_file)

    # # Call the plotting functions
    # plot_accuracies(results, accuracy_fig_path)
    # plot_fidelity_metric_with_ci(results, 'Maximum Fidelity Gap', 'mean_max_fid_gap', 'ci_max_fid_gap', max_fid_gap_fig_path)
    # plot_fidelity_metric_with_ci(results, 'Mean Fidelity Gap (AUROC)', 'mean_fid_gap_auroc', 'ci_fid_gap_auroc', fid_gap_auroc_fig_path)
    # plot_fidelity_metric_with_ci(results, 'Mean Fidelity Gap (Accuracy)', 'mean_fid_gap_accuracy', 'ci_fid_gap_accuracy', fid_gap_accuracy_fig_path)
    # plot_fidelity_metric_with_ci(results, 'Mean Fidelity Gap (Residual Error)', 'mean_fid_gap_residual_error', 'ci_fid_gap_residual_error', fid_gap_residual_error_fig_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='nn', help='Model to use for classification')
    parser.add_argument('--k', type = int, default=2, help = 'k top feature, use k = 3 adult income data')
    parser.add_argument('--setup', type = int, default=2, help = 'Defining the setup') # setup 2 means no M in the model training 
    parser.add_argument('--sensitive', type=str, default='M', help='Sensitive attribute name')
    args = parser.parse_args()

    main(args)

