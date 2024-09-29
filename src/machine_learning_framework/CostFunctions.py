import numpy as np
import matplotlib.pyplot as plt


class CostFunctionInterface:
    """
    Interface for costfunctions used in thetrainingmodul
    """

    @staticmethod
    def calculate(predictions, targets, weights = [0], lam = 0):
        """
        static function for calculating the loss
        :param np-array of predictions: predictions ( normally calculated by forwardpropagation of neuronal network)
        :param targets: np-array of real values (ground truth)
        :return: loss value
        """
        pass

class CrossEntropy(CostFunctionInterface):
    """
    Class for calculating the cross entropy
    """

    @staticmethod
    def calculate(predictions, targets, weights = [0], lam = 0) -> float:
        """
        static function for calculating the cross entropy
        :param np-array of predictions: predictions ( normally calculated by forwardpropagation of neuronal network)
        :param targets: np-array of real values (ground truth)
        :return: cross entropy
        """
        predictions_clipped = np.clip(predictions, 0.000000001, 0.99999999)
        weight_pow_sum = sum([np.sum(weight**2) for weight in weights])
        return np.mean(- targets * np.log(predictions_clipped) - (1 - targets) * np.log(1 - predictions_clipped)) + 1/(2 * len(predictions_clipped)) * lam * weight_pow_sum


class MeanSquaredError(CostFunctionInterface):
    """
    Class for calculating the MSE
    """

    @staticmethod
    def calculate(predictions, targets, weights = [0], lam = 0) -> float:
        """
        static function for calculating the MSE
        :param np-array of predictions: predictions ( normally calculated by forwardpropagation of neuronal network)
        :param targets: np-array of real values (ground truth)
        :return: MSE
        """
        test = np.mean((predictions - targets) ** 2)
        weight_pow_sum = sum([np.sum(weight**2) for weight in weights])
        return np.mean((predictions - targets) ** 2) + 1/(2 * len(predictions)) * lam * weight_pow_sum


class CategoricalCrossEntropy(CostFunctionInterface):
    """
    Class for calculation the CCE
    """

    @staticmethod
    def calculate(h, y_one_hot, weights = [0], lam = 0):
        """
        static function for calculating the CCE
        :param h: prediction in form of a nparray
        :param y_one_hot: One Hot vector of the truth
        :return:
        """
        h = np.clip(h, a_min=0.000000001, a_max=None)
        error_per_sample = np.sum(y_one_hot * np.log(h), axis=1)
        weight_pow_sum = sum([np.sum(weight**2) for weight in weights])
        return -1 * np.sum(error_per_sample / len(error_per_sample)) + 1/(2 * len(error_per_sample)) * lam * weight_pow_sum


    @staticmethod
    def calculate_single_sample(h, y_one_hot):
        h = np.clip(h, a_min=0.000000001, a_max=None)
        test = np.log(h)
        error_per_sample = y_one_hot * np.log(h)
        return -1 * error_per_sample
