from sklearn import metrics
# import numpy as np

# def accuracy(truth, pred):
#     return metrics.accuracy_score(truth, pred)

# def precision(y_true, y_pred):
#     true_positives = np.sum(np.logical_and(y_true == 1, y_pred == 1))
#     false_positives = np.sum(np.logical_and(y_true == 0, y_pred == 1))
#     precision = true_positives / (true_positives + false_positives + 1e-9)
#     return precision

# def recall(y_true, y_pred):
#     true_positives = np.sum(np.logical_and(y_true == 1, y_pred == 1))
#     false_negatives = np.sum(np.logical_and(y_true == 1, y_pred == 0))
#     recall = true_positives / (true_positives + false_negatives + 1e-9)
#     return recall

# def f1score(y_true, y_pred):
#     p = precision(y_true, y_pred)
#     r = recall(y_true, y_pred)
#     f1score = 2 * (p * r) / (p + r + 1e-9)
#     return f1score 

def accuracy(truth, pred):
    return metrics.accuracy_score(truth, pred)

def precision(y_true, y_pred):
    return metrics.precision_score(y_true, y_pred)

def recall(y_true, y_pred):
    return metrics.recall_score(y_true, y_pred)

def f1score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred)
