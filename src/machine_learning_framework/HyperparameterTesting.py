from src.machine_learning_framework.TrainingModel import *
def train_multiple_hyperparameters(input_data, truth_data, validation_data, validation_truth, layer, epoch,
                                   learning_rate, lam, abs_path, costfunction = CategoricalCrossEntropy(), optimizer = GradientDescentOptimizer()):
    """
    This method trains models with different hyperparameters and prints several metrics. The neuronal network always contains hidden layers and a SoftmaxLayer as last layer.
    :param input_data: the input data
    :param truth_data: the truth (target output for each sample)
    :param validation_data: the validation data
    :param validation_truth: the truth of the validation data
    :param layer: list of arrays containing the size of the neuronal network for each hyperparameter test. For each test, we need one array in the list.
                len(layer[i]) equals the number of layers in the model of the i-th testing, layer[i][j] equals the number of neurones in layer j in the i-th testing
    :param epoch: list of integers representing the number of epochs in each hyperparameter test
    :param learning_rate: list of floats representing the learning rate (alpha) in each hyperparameter test
    :param lam: list of floats representing the regularization parameter (lambda)
    :param abs_path: absolut path, where the textfile containing the metrics will be saved
    :return: dictionary with informations of the winner
    """
    winner = {}
    winner_f1_score = 0

    # check given data
    if (not (len(layer) == len(epoch) == len(learning_rate) == len(lam))):
        print(
            "Error: the length of layers, epochs, learning rate and lambdas must match. Otherwise we can't test all hyperparameters!")
        return

    number_of_tests = len(layer)
    # for each given set of hyperparameters
    for test in range(number_of_tests):

        # create layers of given size
        list_of_layer = []
        # first layer has always input size equal to number of inputs
        current_layer = ClassicLayer(len(input_data[0]), layer[test][0])
        list_of_layer.append(current_layer)
        # crate all layers in between the input and the output layer
        for layer_index in range(1, len(layer[test]) - 1):
            current_layer = ClassicLayer(layer[test][layer_index - 1], layer[test][layer_index])
            list_of_layer.append(current_layer)
        # create output layer
        current_layer = SoftmaxLayer(layer[test][-2], layer[test][-1])
        list_of_layer.append(current_layer)

        # create neuronal network model
        current_model = NeuronalNetworkModel(list_of_layer)

        # create training model
        current_training_model = TrainingModel(current_model, costfunction, optimizer)
        print(f"Models of test {test + 1} created.")

        # train model
        error_history = current_training_model.train(input_data, truth_data, epoch[test], learning_rate[test], lam[test] ,prints=True, graphics=False)
        print(f"Training of test {test+1} finished.")

        # classify validation data
        prediction_validation_data = current_model.classification(validation_data)

        # create plot
        plt.plot(error_history, label="")
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title(f'Error of test {test+1}')
        file_path_png = os.path.join(abs_path, f"plot_test{test + 1}.png")
        plt.savefig(file_path_png)
        plt.close()

        # write textfile
        file_path_txt = os.path.join(abs_path, f"log_hyperparameter_testing.txt")
        with open(file_path_txt, 'a') as f:
            f.write(f"\n \n \n ##################################################### \n"
                    f"##################################################### \n"
                    f"Starting test {test + 1} of {number_of_tests} \n")
            f.write(f"Layers: \t\t{layer[test]} \n"
                    f"epoch: \t\t\t{epoch[test]} \n"
                    f"learning rate: \t{learning_rate[test]} \n"
                    f"lambda: \t\t{lam[test]} \n")
            f.write(f"Final error: \t{error_history[-1]} \n")
            f.write(f" \n ####################### Metrics test {test+1} ################### \n"
                    f"Overall Accuracy: \t\t\t {Metrics.accuracy(prediction_validation_data, validation_truth)} \n"
                    f"Accuracy per class: \t\t {Metrics.accuracy_per_class(prediction_validation_data, validation_truth)} \n"
                    f"Macro average accuracy: \t {Metrics.macro_average_accuracy(prediction_validation_data, validation_truth)} \n"
                    f"F1 score per class: \t\t {Metrics.f1_score_per_class(prediction_validation_data, validation_truth)} \n"
                    f"F1 score average: \t\t\t {Metrics.f1_score_average(prediction_validation_data, validation_truth)} \n")

        if Metrics.f1_score_average(prediction_validation_data, validation_truth) > winner_f1_score:
            winner = {'layer': layer[test], 'epoch': epoch[test], 'learning_rate': learning_rate[test], 'lam': lam[test]}
            winner_f1_score = Metrics.f1_score_average(prediction_validation_data, validation_truth)

    file_path_txt = os.path.join(abs_path, f"log_hyperparameter_testing.txt")
    with open(file_path_txt, 'a') as f:
        f.write(f" \n\n ###################### \n"
                f"Winner is {winner}\n")

    return winner
def randomized_search(input_data, truth_data, validation_data, validation_truth, layer_sizes, numbers_neuron, learning_rates, epochs, lambdas, number_of_test, number_of_classes, abs_path):

    """
    This method trains models with different hyperparameters and prints several metrics. The neuronal network always contains hidden layers and a SoftmaxLayer as last layer.
    The hyperparameters are chosen randomly out of the values in the given lists.
    :param input_data: the input data
    :param truth_data: the truth (target output for each sample)
    :param validation_data: the validation data
    :param validation_truth: the truth of the validation data
    :param layer_sizes: a list of all possible number of layers
    :param numbers_neuron: a list containing all possible numbers of neurons. Each layer will have rhe same number of neurons
    :param learning_rates: a list of possible learning rates
    :param epochs: a list of possible epochs
    :param lambdas: a list containing all possible lambdas
    :param number_of_test: the number of tests the function should do
    :param number_of_classes: the number of output classes (output of the last layer)
    :return: nothing
    """

    list_layers = [None] * number_of_test
    list_epochs = [None] * number_of_test
    list_learning_rates = [None] * number_of_test
    list_lambdas = [None] * number_of_test
    for i in range(number_of_test):
        layer_size = layer_sizes[np.random.randint(0, len(layer_sizes))]
        neurones = numbers_neuron[np.random.randint(0, len(numbers_neuron))]
        layer = [neurones] * (layer_size -1) + [number_of_classes]
        list_layers[i] = layer

        learning_rate = learning_rates[np.random.randint(0, len(learning_rates))]
        list_learning_rates[i] = learning_rate

        epoche = epochs[np.random.randint(0, len(epochs))]
        list_epochs[i] = epoche

        lam = lambdas[np.random.randint(0, len(lambdas))]
        list_lambdas[i] = lam

    winner = train_multiple_hyperparameters(input_data, truth_data, validation_data, validation_truth, list_layers, list_epochs, list_learning_rates, list_lambdas, abs_path)
