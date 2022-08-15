import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    TP = np.sum(np.logical_and(prediction, ground_truth))       
    precision = TP/np.sum(prediction)
    recall = TP/np.sum(ground_truth)
    accuracy = np.sum(prediction == ground_truth)/prediction.size
    f1 = precision * recall / (precision + recall)

    # implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # computing accuracy
    result_bool = (prediction==ground_truth)
    accuracy = np.sum(result_bool)/result_bool.shape[0]
    return accuracy
