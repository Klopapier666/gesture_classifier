import numpy as np
import matplotlib.pyplot as plt
from src.machine_learning_framework.NeuronalNetworkLayer import *

class NeuronalNetworkModel:
    """
    This class represents the neural network. It contains following members:
    - layers_list: sorted list with all layers (first layer is the input layer, last one the output layer)
    """

    def __init__(self, list_of_layers):
        self.layers_list = list_of_layers


    def get_layers_list(self):
        return self.layers_list

    def add_layer(self, layer):
        self.layers_list.append(layer)
        return layer

    def forward(self, sample):
        """
        This method takes a sample and propargates it forward through all layers
        :param sample: the given sample (input for the input layer)
        :return: list of the outputs of each layer
        """

        input = sample.copy()
        # list wich contains the input at first position and all outputs.
        list_of_outputs = [None] * (len(self.layers_list)+1)
        list_of_outputs[0] = input

        for index, layer in enumerate(self.layers_list):
            list_of_outputs[index+1] = layer.forward(input)
            input = list_of_outputs[index+1]



        return list_of_outputs


    def backward(self, list_of_outputs, truth):
        """
        This method determines the partial derivatives according to the weights of all layers.
        The method uses a lamda as regularization parameter.
        :param list_of_outputs: this list containes the outputs of each layer
        :param truth: the truth for the treated sample
        :return: list of partial derivatives of weights and bias per layer
        """

        # check of we have the outputs of all layers
        if (len(list_of_outputs)-1 != len(self.layers_list)):
            print("Error during backwardpropagation!The number of outputs has to be the same as the number of layers!")

        # these lists contain the derivative per layer.
        derivative_weight_list = [None] * (len(self.layers_list))
        derivative_bias_list = [0] * (len(self.layers_list))

        # compute the delta of the output layer
        #delta_previous_layer = self.layers_list[-1].backward(truth, list_of_outputs[-1])
        delta_previous_layer = list_of_outputs[-1] - truth

        # backwards iterate over all layers.
        for index in range(len(self.layers_list) - 1, -1, -1):
            # propagate backwards through this layer
            delta_previous_layer, derivative_weight_list[index], derivative_bias_list[index] = self.layers_list[index].backward(delta_previous_layer, list_of_outputs[index])

        return derivative_weight_list, derivative_bias_list

    def classification(self, data):
        """
        This function is used to classify the data. It just returns the class, the model finds most probable.
        :param data: data to classify
        :return: the clasified data
        """
        classification = np.zeros((data.shape[0], self.get_layers_list()[-1].get_output_size()))


        for index, sample in enumerate(data):
            classification[index] = self.forward(sample)[-1]

        argmax = np.argmax(classification, axis=1)

        return argmax

