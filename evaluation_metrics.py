import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from scipy.stats import wilcoxon, kruskal, mannwhitneyu
from itertools import combinations

## overall_preds is prediction from lime 
class EvaluationMetrics:
    def __init__(self, model, X_test, y_true, indices_dict, w, k, exps, overall_preds, group_preds, group_explanation_preds_proba):
        self.model = model
        self.X_test = X_test
        self.y_true = y_true
        self.indices_dict = indices_dict
        self.w = w
        self.k = k
        self.exps = exps
        self.overall_preds = overall_preds
        self.group_preds = group_preds
        self.ml_overall_preds = self.model.predict(self.X_test)
        self.group_explanation_preds_proba = group_explanation_preds_proba

    def ground_truth_fidelity(self, group='all'):
        if group == 'all':
            indices = list(range(len(self.exps)))
        else:
            indices = self.indices_dict[group]

        fidelity_scores = []
        for i in indices:
            exp = self.exps[i]
            fidelity_score = self._ground_truth_single(self.w, self.k, exp)
            fidelity_scores.append(fidelity_score)

        return np.mean(fidelity_scores)

    def _ground_truth_single(self, w, k, exp):
        feat = np.argsort(np.abs(w))[::-1]
        best_k = feat[:k]

        exp_feat = np.argsort(np.abs(exp))[::-1]
        best_k_exp = exp_feat[:k]

        overlap = set(best_k).intersection(set(best_k_exp))

        return len(overlap) / k

    def maximum_fidelity_gap(self):
        overall_accuracy = accuracy_score(self.ml_overall_preds, self.overall_preds)
        fidelity_gap_dict = {}
        
        for group, indices in self.indices_dict.items():
            group_accuracy = accuracy_score(self.ml_overall_preds[indices], self.group_preds[group])
            fidelity_gap_dict[group] = overall_accuracy - group_accuracy
            
        max_fidelity_gap_group = max(fidelity_gap_dict, key=fidelity_gap_dict.get)
        max_fidelity_gap = fidelity_gap_dict[max_fidelity_gap_group]
        return max_fidelity_gap, fidelity_gap_dict, max_fidelity_gap_group

    def mean_fidelity_gap_auroc(self):
        auroc_dict = {}
        for group, indices in self.indices_dict.items():
            if len(np.unique(self.ml_overall_preds[indices])) == 1:
                auroc_dict[group] = 0
            else:
                auroc_dict[group] = roc_auc_score(self.ml_overall_preds[indices], self.group_explanation_preds_proba[group])

        auroc_gaps = []
        for group1, group2 in combinations(auroc_dict.keys(), 2):
            auroc_gaps.append(np.abs(auroc_dict[group1] - auroc_dict[group2]))
        
        auroc_gap = np.mean(auroc_gaps)

        return auroc_gap


    def mean_fidelity_gap_accuracy(self):
        accuracy_dict = {}
        for group, indices in self.indices_dict.items():
            accuracy_dict[group] = accuracy_score(self.ml_overall_preds[indices], self.group_preds[group])

        accuracy_gaps = []
        for group1, group2 in combinations(accuracy_dict.keys(), 2):
            accuracy_gaps.append(np.abs(accuracy_dict[group1] - accuracy_dict[group2]))
        
        #print(f'accuracy_gap {accuracy_gaps}')
        
        accuracy_gap = np.mean(accuracy_gaps)

        return accuracy_gap


    def mean_fidelity_gap_residual_error(self):
        residual_error_dict = {}
        for group, indices in self.indices_dict.items():
            residual_error_dict[group] = np.mean(np.abs(self.ml_overall_preds[indices] - self.group_preds[group]))

        error_gaps = []
        for group1, group2 in combinations(residual_error_dict.keys(), 2):
            error_gaps.append(np.abs(residual_error_dict[group1] - residual_error_dict[group2]))
        
        error_gap = np.mean(error_gaps)

        return error_gap
    
    def mannwhitneyu_p_value(self, male_fidelity_list, female_fidelity_list):
        u_statistic, p_value = mannwhitneyu(male_fidelity_list, female_fidelity_list)
        return p_value
    
    def kruskal_test(self):
        # Create a list to hold fidelity values for each group
        group_fidelities = []

        # Iterate over each group
        for group in self.indices_dict.keys():
            # Calculate fidelity for this group
            fidelity = self.ground_truth_fidelity(group)
            # Append the calculated fidelity to the list
            group_fidelities.append(fidelity)
        
        # Use the kruskal function from scipy.stats to compute the Kruskal-Wallis H test
        H_statistic, p_value = kruskal(*group_fidelities)
        
        return p_value

  
    def one_sided_wilcoxon_signed_rank_test(self, data_diff):
        w_greater, p_greater = wilcoxon(data_diff, y=None, alternative='greater')
        print("Wilcoxon signed-rank test (greater) using differences: W = {}, p-value = {}".format(w_greater, p_greater))

        w_less, p_less = wilcoxon(data_diff, y=None, alternative='less')
        print("Wilcoxon signed-rank test (less) using differences: W = {}, p-value = {}".format(w_less, p_less))

    def discrepancies_between_ml_and_lime(self):
        discrepancies = self.overall_preds != self.ml_overall_preds
        discrepancy_indices = np.where(discrepancies)[0]
        return discrepancy_indices