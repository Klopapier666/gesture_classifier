import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from src.machine_learning_framework.NeuronalNetworkModel import *
from src.machine_learning_framework.CostFunctions import *
from src.machine_learning_framework.Optimizer import *
import src.machine_learning_framework.Metrics as Metrics

class TrainingModel:
    """This class is used for training a Neuronal Network
    It containes a NeuronalNetworkModel, a cost function and an optimizer"""

    def __init__(self, model: object, cost_function: object, optimizer: object) -> object:
        self.model = model
        self.cost_function = cost_function
        self.optimizer = optimizer

    def backpropagation(self, list_of_lists_of_outputs, list_of_truths, lam):
        """ This method determines the mean of all partial derivatives according to the weights of all layers.
        The method uses ta lamda as regularization parameter.
        :param list_of_lists_of_outputs: for each sample this list contains a list with the outputs of each layer
        :param list_of_truths: list containing the truth for each sample in list_of_lists_of_outputs
        :param lam: the regularization parameter
        :return: list of partial derivatives of weights and bias per layer
        """

        # these lists contain a value per layer (except the output layer, because this doesn't have weights)
        # the value i is the sum of the derivatives according to each sample
        derivative_weight_sum_list = [0] * (len(self.model.get_layers_list()))
        derivative_bias_sum_list = [0] * (len(self.model.get_layers_list()))

        # check, if we have the truth for each output
        if (len(list_of_lists_of_outputs) != len(list_of_truths)):
            print(
                "Error in backwardpropagation! The size of the set of outputs has to be the same as the size of the set of truths")

        # for each sample
        test_counter = 0
        for list_of_outputs, truth in zip(list_of_lists_of_outputs, list_of_truths):
            # backwardspropagation through the neuronal network
            derivative_weight_list, derivative_bias_list = self.model.backward(list_of_outputs, truth)
            # sum up the derivatives for building the mea afterwards
            derivative_weight_sum_list = [x + y for x, y in zip(derivative_weight_sum_list, derivative_weight_list)]
            derivative_bias_sum_list = [x + y for x, y in zip(derivative_bias_sum_list, derivative_bias_list)]
            test_counter +=1

        # for each layer build the mean of the derivatives
        derivatives_weight = [
            (derivative_weight_sum_element / len(list_of_lists_of_outputs)) + (lam/ len(list_of_lists_of_outputs)) * layer.get_weights() for
            derivative_weight_sum_element, layer in
            zip(derivative_weight_sum_list, self.model.get_layers_list())]
        derivatives_bias = [(derivative_bias_sum_element) / len(list_of_lists_of_outputs) for
                            derivative_bias_sum_element in derivative_bias_sum_list]

        return derivatives_weight, derivatives_bias

    def train(self, input_data, truth_data, epoch, learning_rate, lam, abs_path="", graphics=False, prints = False, save_weights = False, model_name = ''):
        """
        This method does the real training
        :param self:
        :param input_data: the input data
        :param truth_data: the truth (target output for each sample, integer)
        :param learning_rate: the learning rate (alpha)
        :param lam: the regularization parameter (lambda)
        :param abs_path: the path to the directory, where the weights should be saved.
        :return: array of error of each epoch
        """

        truth_one_hot = self.one_hot_encoding(truth_data)
        error_history = [0] * epoch

        for epoch in range(epoch):

            # forwardpropagation
            list_of_lists_of_outputs = [None] * len(input_data)
            predictions_list = np.zeros((len(input_data), self.model.get_layers_list()[-1].get_output_size()))

            for index, (sample, list_of_outputs, prediction) in enumerate(zip(input_data, list_of_lists_of_outputs, predictions_list)):
                list_of_lists_of_outputs[index] = self.model.forward(sample)
                predictions_list[index] = list_of_lists_of_outputs[index][-1]

            if(prints):
                print("\n ##################### \n")
                print(f"Epoch: {epoch} \t Forwardpropagation finished.")
                #print(f"Outputs: \t {list_of_lists_of_outputs}")

            # save error
            weights = [layer.get_weights() for layer in self.model.get_layers_list()]
            error_history[epoch] = self.cost_function.calculate(predictions_list, truth_one_hot, weights, lam)

            if(prints):
                print(f"Epoch {epoch} \t Loss: {error_history[epoch]}.")
                test = predictions_list.round()
                test = np.array(predictions_list.round() == truth_one_hot)
                print(f"Epoch {epoch} \t Accuracy: {np.mean(predictions_list.round() == truth_one_hot)}")

            # backpropagation
            derivatives_weight, derivatives_bias = self.backpropagation(list_of_lists_of_outputs, truth_one_hot, lam)

            if(prints):
                print(f"Epoch: {epoch} \t Backpropagation finished.")
                #print(f"Derivatives weight: {derivatives_weight}")
                #print(f"Derivatives bias: {derivatives_bias}")

            # update weights
            for index, (layer, derivative_weight, derivative_bias) in enumerate( zip(self.model.get_layers_list(), derivatives_weight, derivatives_bias)):
                self.optimizer.update_weights(self.model.get_layers_list()[index], learning_rate, derivative_weight, derivative_bias)


            if(prints):
                #for index, layer in enumerate(self.model.get_layers_list()):
                    #print(f"Epoch: {epoch} \t optimized weight layer {index}: {layer.get_weights()}")
                    #print(f"Epoch: {epoch} \t optimized bias layer {index}: {layer.get_bias()}")
                print(f"Epoche {epoch} finished.")

            if (graphics and (epoch+1) % 10 == 0):
                plt.plot(error_history[:epoch])
                plt.show()

        if(save_weights):
            # save weights
            weights = []
            biases = []
            for layer in self.model.get_layers_list():
                weights.append(layer.get_weights())
                biases.append(layer.get_bias())

            file_weights_abs = os.path.join(abs_path, f'saved_weights_{model_name}.pkl')

            with open(file_weights_abs, 'wb') as datei:
                pickle.dump((weights, biases), datei)

        list_of_lists_of_outputs = [None] * len(input_data)
        predictions_list = np.zeros((len(input_data), self.model.get_layers_list()[-1].get_output_size()))

        for index, (sample, list_of_outputs, prediction) in enumerate(zip(input_data, list_of_lists_of_outputs, predictions_list)):
            list_of_lists_of_outputs[index] = self.model.forward(sample)
            predictions_list[index] = list_of_lists_of_outputs[index][-1]

        #np.savetxt(os.path.join(os.getcwd(), f'resource/testing/prediction_trainingdata_22_03.txt'), np.array(predictions_list))
        #layer_weights_list = [layer.get_weights() for layer in self.model.get_layers_list()]
        #with open(os.path.join(os.getcwd(), f'resource/testing/weight_trainingdata_22_03.txt'), 'w') as f:
        #    f.write(f"{layer_weights_list}")

        return error_history

    def one_hot_encoding(self, y):
        """
        Encodes categorical features into one-hot vectors
        :param X: Training data
        :param y: truth to be encoded
        :return: X and encoded truth
        """
        y = y.astype(int)
        classes = np.unique(y)
        y_one_hot = np.zeros((len(y), len(classes)))
        y_one_hot[np.arange(len(y)), y] = 1
        return y_one_hot
