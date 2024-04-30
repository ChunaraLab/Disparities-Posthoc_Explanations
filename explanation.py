import numpy as np
from lime import lime_tabular

class LimeExplanations:
    def __init__(self, X_train, cat_inds, X_test, pred_proba, n_samples=1000, random_state=None):
        self.X_train = X_train
        self.cat_inds = cat_inds
        self.X_test = X_test
        self.pred_proba = pred_proba
        self.n_samples = n_samples
        self.explainer = lime_tabular.LimeTabularExplainer(X_train,
                                                           discretize_continuous=False,
                                                           sample_around_instance=True,
                                                           categorical_features=cat_inds)
        self.explanation_objects = self.get_explanation_objects()
        self.explanations = self.get_explanations()

    def get_explanation_objects(self):
        explanation_objects = []
        for x in self.X_test:
            exp_obj = self.explainer.explain_instance(
                x,
                predict_fn=self.pred_proba,
                num_samples=self.n_samples,
                num_features=self.X_train.shape[1],  # Set num_features to the total number of features in your dataset
            )
            explanation_objects.append(exp_obj)
        return explanation_objects

    def get_explanations(self):
        explanations = []
        for exp_obj in self.explanation_objects:
            exp = exp_obj.local_exp[1]
            exp = sorted(exp, key=lambda tup: tup[0])
            exp = [t[1] for t in exp]
            explanations.append(exp)
        return explanations

    def get_overall_predictions(self):
        overall_predictions = []
        for exp_obj in self.explanation_objects:
            local_prediction = exp_obj.local_pred[0]  # Get the local prediction from the explanation object
            pred = 1 if local_prediction >= 0.5 else 0  # Classify label based on the local prediction
            overall_predictions.append(pred)
        return np.array(overall_predictions)


    def get_group_predictions(self, indices_dict):
        group_preds = {}
        overall_preds = self.get_overall_predictions()
        for category, indices in indices_dict.items():
            group_preds[category] = overall_preds[indices]
        return group_preds
    

    def get_group_explanation_proba(self, indices_dict):
        group_proba = {}
        overall_proba = self.get_overall_proba()
        for category, indices in indices_dict.items():
            group_proba[category] = overall_proba[indices]
        return group_proba

    def get_overall_proba(self):
        overall_proba = []
        for exp_obj in self.explanation_objects:
            local_prediction = exp_obj.local_pred[0]  # Get the local prediction from the explanation object
            overall_proba.append(local_prediction)
        return np.array(overall_proba)