import numpy as np
import matplotlib.pyplot as plt
from src.machine_learning_framework.ActivationFunctions import *

# Anmerkung: Die Gewchte gehören nicht in wie in der VL (neural_network_diagram_solution.pdf) zum Layer links der Schnittstelle, sondern zum Layer rechts der Schnittstelle.
# D.h. ein Layer berechnet erst anhand der gewichte das z, dann das a und gibt das a zurück. Das muss in der backpropagation beachtet werden.


class NNLayerInterface:
    """This Interface defines a neural network layer"""

    def __init__(self, input_size, output_size, weights = None, bias = None):
        self.input_size = input_size
        self.output_size = output_size
        if(weights is None or bias is None):
            self.weights = np.random.uniform(-1, 1, (output_size, input_size))
            self.bias = np.random.uniform(-1, 1, output_size)
        else:
            self.weights = weights
            self.bias = bias




    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def set_weights(self, weights):
        self.weights = weights.copy()

    def set_bias(self, bias):
        self.bias = bias.copy()

    def set_weights_bias(self, weights, bias):
        self.weights = weights.copy()
        self.bias = bias.copy()

    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.output_size

    def forward(self, input)->np.array:
        """Forward pass of the neural network layer for given input"""
        pass

    def backward(self, prediction, truth):
        """Backward pass of the neural network layer for given prediction and truth"""
        pass

class ClassicLayer(NNLayerInterface):
    """
    This class defines a Hidden Layer with the sigmondi function as activation function.
    """

    def __init__(self, input_size, output_size, weights = None, bias = None):
        """
        Constructor
        :param input_size: outputsize of previous layer
        :param output_size: outputsize of this layer (number of neurons)
        :param weights: matrix of form outputsize x input_size
        :param bias: array of dimension outputsize
        """
        super().__init__(input_size, output_size, weights, bias)

    def forward(self, input) -> np.array:
        """
        Forward pass of the neuronal network layer for given input
        :param input: given input as np-array
        :return: output as 2D matrix
        """

        # checks, if the input has correct size
        if(not input.ndim == 1):
            print("Error during foorwardpropagation. The given input has to be a 1D np-array")
        if(input.shape[0] != self.input_size):
            print("Error during forwardpropagation. The given input has to be the size of the current layer")

        # determine the weighted input for each neuron
        z = self.weights @ input + self.bias
        # determine activation values
        a = ActivationFunctions.sigmoid(z)
        # return output of the layer
        return a

    def backward(self, delta_previous_layer, input_this_layer):
        """
        Backward pass of the neuronal network layer
        :param delta_previous_layer: the delta of the previous layer (layer closer to output of neuronal network. Previous in sense of direction of backpropagation.)
        :param input_this_layer: the output (a) of this layer during the forewardpropagation
        :return: delta of this layer and partial derivative_weight ( = a * delta_this_layer) and derivative_bias of this layer
        """

        # determine the delta of this layer (https://lectures.hci.informatik.uni-wuerzburg.de/ws23/ml/06-02-backpropagation-deck.html#/computing-the-gradients-error-and-backpropagation-4)
        delta_this_layer = (self.weights.T @ delta_previous_layer) * input_this_layer * (1 - input_this_layer)

        # determination of the partial derivation of the cost function according to the weights (with the exception of the bias weights)
        derivative_weight = delta_previous_layer[:, np.newaxis] @ input_this_layer[:, np.newaxis].T

        ## determination of the partial derivation of the cost function according to the bias weights
        derivative_bias = delta_previous_layer

        return delta_this_layer, derivative_weight, derivative_bias

class SoftmaxLayer(NNLayerInterface):
    """defines the outputlayer inclusive the softmaxlayer. They are combined in one layer so that we can handle the special cases in the output layer"""


    def forward(self, input):
        """
        Forward pass of the neuronal network layer for given input
        :param input: given input as np-array
        :return: output as np.array
        """

        # checks, if the input has correct size
        if (not input.ndim == 1):
            print("Error during foorwardpropagation. The given input has to be a 1D np-array")
        if (input.shape[0] != self.input_size):
            print("Error during forwardpropagation. The given input has to be the size of the current layer")

        # forwardpropagation trough outputlayer
        # determine the weighted input for each neuron
        z = self.weights @ input + self.bias
        # no activation function in outputlayer
        a = z

        # calculate softmax
        output = ActivationFunctions.softmax(a)

        return output

    def backward(self, delta_previous_layer, input_this_layer):
        """
        Backward pass of the neuronal network layer
        :param delta_previous_layer: the delta of the previous layer (layer closer to output of neuronal network. Previous in sense of direction of backpropagation.)
        :param input_this_layer: the output (a) of this layer during the forewardpropagation
        :return: delta of this layer and partial derivative_weight ( = a * delta_this_layer) and derivative_bias of this layer
        """

        # determine the delta of this layer (https://lectures.hci.informatik.uni-wuerzburg.de/ws23/ml/06-02-backpropagation-deck.html#/computing-the-gradients-error-and-backpropagation-4)
        delta_this_layer = (self.weights.T @ delta_previous_layer) * input_this_layer * (1 - input_this_layer)

        # determination of the partial derivation of the cost function according to the weights (with the exception of the bias weights)
        derivative_weight = delta_previous_layer[:, np.newaxis] @ input_this_layer[:, np.newaxis].T

        ## determination of the partial derivation of the cost function according to the bias weights
        derivative_bias = delta_previous_layer

        return delta_this_layer, derivative_weight, derivative_bias
