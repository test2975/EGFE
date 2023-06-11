import torch
from sklearn.metrics import f1_score, r2_score, confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sn
import matplotlib.pyplot as plt


def creat_confusion_matrix(y_true, y_pred):
    cf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(7, 7))
    return sn.heatmap(cf_matrix, annot=True).get_figure()


def accuracy_simple(scores, targets):
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    return accuracy_score(targets, scores)


def precision(scores, targets, average='macro'):
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    return precision_score(targets, scores, average=average, zero_division=0)


def recall(scores, targets, average='macro'):
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    return recall_score(targets, scores, average=average, zero_division=0)


def f1score(scores, targets, average='macro'):
    """Computes the F1 score using scikit-learn for binary class labels. 

    Returns the F1 score for the positive class, i.e. labelled '1'.
    """
    targets = targets.cpu().numpy()
    scores = scores.cpu().numpy()
    return f1_score(targets, scores, average=average)


def r2score(scores, targets):
    y_true = targets.cpu().numpy()
    y_pred = scores.cpu().numpy()
    return r2_score(y_true, y_pred)
