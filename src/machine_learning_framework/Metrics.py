import numpy as np
import matplotlib.pyplot as plt
def accuracy(predictions, targets):
    """
    Computes the accuracy
    :param predictions: The calculated output of the model (already classified, only containing integer)
    :param targets:  The truth
    :return: accuracy of the model
    """
    return np.mean(predictions == targets)

def accuracy_per_class(predictions, targets):
    """
    Computes the accuracy per class
    :param predictions: The calculated output of the model (already classified, only containing integer)
    :param targets: the target data, also integer describing the class
    :return: np array with accuracy per class
    """
    classes = np.unique(targets)
    result = np.zeros(len(classes))
    for index, predicted_class in enumerate(classes):
        result[index] = ((predictions == predicted_class) & (targets == predicted_class)).sum()/((targets == predicted_class)).sum()

    return result


def macro_average_accuracy(predictions, targets):
    """
    Computes the
    :param predictions: The calculated output of the model (already classified, only containing integer)
    :param targets: the target data, also integer describing the class
    :return: macro average accuracy of the model
    """
    return np.mean(accuracy_per_class(predictions, targets))


def f1_score(predictions, targets, prediction_class):
    """
    Computes the F1 score for the given class
    :param predictions: The calculated output of the model (already classified, only containing integer)
    :param targets: the target data, also integer describing the class
    :param prediction_class: the class, for wich we want to compute the F1
    :return: f1 score of the model
    """
    true_positives = (predictions == prediction_class) & (targets == prediction_class)
    false_positives = (predictions == prediction_class) & (targets != prediction_class)
    false_negatives = (predictions != prediction_class) & (targets == prediction_class)

    if(true_positives.sum() == 0):
        return 0

    precision = true_positives.sum() / (true_positives.sum() + false_positives.sum())
    recall = true_positives.sum() / (true_positives.sum() + false_negatives.sum())

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1

def f1_score_per_class(predictions, targets):
    """
    Computes the F1 score for each class
    :param predictions: The calculated output of the model (already classified, only containing integer)
    :param targets: the target data, also integer describing the class
    :return: array with F1 scores for each class
    """
    classes = np.unique(targets)
    result = np.zeros(len(classes))
    for index, predicted_class in enumerate(classes):
        result[index] = f1_score(predictions, targets, predicted_class)

    return result

def f1_score_average(predictions, targets):
    """
    Computes the average of the f1 scores for each class
    :pparam predictions: The calculated output of the model (already classified, only containing integer)
    :param targets: the target data, also integer describing the class
    :return: average of all F1 scores
    """
    return np.mean(f1_score_per_class(predictions, targets))