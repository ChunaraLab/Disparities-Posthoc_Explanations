import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BaseModel():
    def train(self, X_train, y_train, verbose=False):
        torch.manual_seed(0)
        X_train = torch.from_numpy(X_train).float().to(device)
        y_train = torch.from_numpy(y_train).float().to(device)

        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        criterion = nn.BCELoss().to(device)
        optimizer = torch.optim.Adam(self.net.parameters(), weight_decay=1e-4)

        # Train model
        epochs = 100
        for ep in range(epochs):
            self.net.train()
            for batch in dataloader:
                X_batch, y_batch = batch[0].to(device), batch[1].to(device)

                optimizer.zero_grad()

                # Forward pass
                y_pred = self.net(X_batch)

                # Compute Loss
                loss = criterion(y_pred[:, 0], y_batch)

                # Backward pass
                loss.backward()

                optimizer.step()
            if verbose:
                print('Epoch {}: train loss: {}'.format(ep, loss.item()))

    def predict_proba(self, X):
        X = torch.from_numpy(np.array(X)).float().to(device)
        class1_probs = self.net(X).detach().cpu().numpy()
        class0_probs = 1 - class1_probs
        return np.hstack((class0_probs, class1_probs))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def evaluate(self, X_test, y_test, indices_dict):
        y_pred = self.predict(X_test)

        accuracies = {}
        for category, indices in indices_dict.items():
            accuracies[category] = accuracy_score(y_test[indices], y_pred[indices])

        overall_accuracy = accuracy_score(y_test, y_pred)
        accuracies['overall'] = overall_accuracy

        return accuracies

    def get_confusion_matrices(self, X_test, y_test, indices_dict):
        y_pred = self.predict(X_test)

        confusion_matrices = {}
        for category, indices in indices_dict.items():
            confusion_matrices[category] = confusion_matrix(y_test[indices], y_pred[indices])

        overall_cm = confusion_matrix(y_test, y_pred)
        confusion_matrices['overall'] = overall_cm

        return confusion_matrices


class LR(BaseModel):
    def __init__(self, num_feat):
        torch.manual_seed(0)
        self.net = nn.Sequential(OrderedDict([
            ('lin', nn.Linear(num_feat, 1)),
            ('last', nn.Sigmoid())
        ])).to(device)


class NN(BaseModel):
    def __init__(self, num_feat):
        torch.manual_seed(0)
        self.net = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(num_feat, 50)),
            ('relu1', nn.ReLU()),
            ('lin2', nn.Linear(50, 100)),
            ('relu2', nn.ReLU()),
            ('lin3', nn.Linear(100, 200)),
            ('relu3', nn.ReLU()),
            ('lin4', nn.Linear(200, 1)),
            ('last', nn.Sigmoid())
        ])).to(device)