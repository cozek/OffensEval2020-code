from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import torch
import pandas as pd
import numpy as np
import torch.nn as nn

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def compute_accuracy(y_pred, y_target):
    y_pred = y_pred.cpu()
    y_target = y_target.cpu()
    return torch.eq(torch.argmax(y_pred,dim=1),y_target).sum().item() / len(y_pred)

def compute_macro_f1(y_pred, y_target, average = 'macro'):
    y_pred = (torch.argmax(y_pred,dim=1)).cpu().long().numpy()
    y_target = y_target.cpu().numpy()

    return f1_score(y_true = y_target, y_pred=y_pred , average=average)


def analyse_preds(y_pred, y_target, threshold=0.5):
    y_pred = (torch.argmax(y_pred,dim=1) > threshold).cpu().long().numpy()
    # y_pred = (torch.argmax(y_pred > threshold,dim=1)).cpu().long().numpy()
    y_target = y_target.cpu().numpy()

    conmat = confusion_matrix(y_pred=y_pred, y_true=y_target)
    confusion = pd.DataFrame(
        conmat, index=["NOT", "HS"], columns=["predicted_NOT", "predicted_HS"]
    )
    print("acc = ", accuracy_score(y_pred=y_pred, y_true=y_target))
    print(classification_report(y_pred=y_pred, y_true=y_target, digits=4))
    print(confusion)

def analyse_preds2(y_pred, y_target, threshold=0.5):
    # y_pred = (torch.argmax(y_pred,dim=1) > threshold).cpu().long().numpy()
    y_pred = torch.argmax(nn.Sigmoid()(y_pred) > threshold,dim=1).cpu().long().numpy()
    y_target = y_target.cpu().numpy()

    conmat = confusion_matrix(y_pred=y_pred, y_true=y_target)
    confusion = pd.DataFrame(
        conmat, index=["NOT", "HS"], columns=["predicted_NOT", "predicted_HS"]
    )
    print("acc = ", accuracy_score(y_pred=y_pred, y_true=y_target))
    print(classification_report(y_pred=y_pred, y_true=y_target, digits=4))
    print(confusion)
    