import numpy as np
import matplotlib.pyplot as plt


class ActivationFunctions:
    """
    Class, which contains all the activation functions as static functions
    """

    @staticmethod
    def sigmoid(z):
        """
        Computes the value of the sigmoid function
        :param z:y input in form of a float/integer or np-array
        :return: value of the sigmoid function
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softmax_multiple_samples(z):
        """
        Computes the softmax of the input
        :param z: input in form of a 2D matrix
        :return: softmax of input as 2D matrix
        """
        # make sure result is a flat and not integer
        result = z.copy().astype(float)

        # for each sample (one row = one sample)
        for row_index in range(result.shape[0]):
            row = result[row_index, :].copy()
            sum_row = np.sum(np.exp(row))
            result[row_index, :] = np.exp(row) / sum_row

        return result

    @staticmethod
    def softmax(z):
        """
        Computes the softmax of the input
        :param z: input in form of a nparray
        :return: softmax of input as nparray
        """
        return np.exp(z) / np.sum(np.exp(z))