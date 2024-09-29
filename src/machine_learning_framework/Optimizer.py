import numpy as np
import matplotlib.pyplot as plt

class OptimizerInterface:
    """Interface class for all optimizers"""

    @staticmethod
    def updateWeights(layer, learning_rate, gradient_weights, gradient_bias):
        """Updates the weights of the given layer with the given gradient"""
        pass

class GradientDescentOptimizer(OptimizerInterface):
    """static class that implements gradient descent"""

    @staticmethod
    def update_weights(layer, learning_rate, gradient_weight, gradient_bias):
        """Updates the weights of the given layer with the given gradient using the gradient descent method"""
        test = layer.get_weights() - learning_rate * gradient_weight
        layer.set_weights(layer.get_weights() - learning_rate * gradient_weight)
        layer.set_bias(layer.get_bias() - learning_rate * gradient_bias)
