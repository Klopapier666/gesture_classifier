import numpy as np
from src.pipeline.interface_pipeline_component import PipelineComponent
from src.machine_learning_framework.TrainingModel import *
import seaborn as sns

class ModelTrainer(PipelineComponent):
    def __init__(self, nn_model, iterations, learning_rate, lam, cost_function = CategoricalCrossEntropy(), optimizer = GradientDescentOptimizer()):
        super().__init__()
        self.input_type = (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        self.output_type = None
        self.iterations = iterations
        self.lam = lam
        self.learning_rate = learning_rate
        self.nn_model = nn_model
        self.nn_training_model = TrainingModel(self.nn_model, cost_function, optimizer)


    def run(self, input_data):
        """
        Run method
        """
        abs_path = os.path.join(os.getcwd(), 'resource/models/final_model_all')
        self.nn_training_model.train(input_data[0], input_data[1], self.iterations, self.learning_rate, self.lam, abs_path,True, True, True, self.model_name)

        ############## TESTING TRAINING##############
        classification_training = self.nn_model.classification(input_data[0])

        correct_predictions = np.zeros(len(classification_training))

        correct_predictions[classification_training == input_data[1]] = 1

        print(f" Accuracy: {np.sum(correct_predictions) / len(correct_predictions)}")

        confusion = np.zeros((8, 8))

        for i in range(8):
            for j in range(8):
                current_matrix = np.zeros(len(classification_training))
                current_matrix[(classification_training == j) & (input_data[1] == i)] = 1
                confusion[i][j] = np.sum(current_matrix)

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        sns.heatmap(confusion,
                    annot=True,  # print pixel values onto each pixel
                    fmt=".0f",  # don't print decimals
                    cmap="gray",  # these are grayscale images, so "gray" is an appropriate colormap
                    ax=ax
                    # we have already initialised a matplotlib ax object we would like to use, so we provide it here
                    );
        ax.set_title("Trainingsdata: Confusion matrix")
        fig.show()
        sum = 0

        for i in range(8):
            print("Trainingsdata: Accuracy of " + str(i) + " is " + str(confusion[i][i] / np.sum(confusion[i])))
            sum += confusion[i][i]

        print("Trainingsdata: Overall accuracy is " + str(sum / np.sum(confusion)))

        print(f"Trainingsdata F1 score per class: {Metrics.f1_score_per_class(classification_training, input_data[1])}")
        print(f"Trainingsdata F1 score: {Metrics.f1_score_average(classification_training, input_data[1])}")


        #################### TESTING VALIDATION ################
        classification_validation = self.nn_model.classification(input_data[2])

        correct_predictions = np.zeros(len(classification_validation))

        correct_predictions[classification_validation == input_data[3]] = 1

        print(f" Accuracy: {np.sum(correct_predictions) / len(correct_predictions)}")

        confusion = np.zeros((8, 8))

        for i in range(8):
            for j in range(8):
                current_matrix = np.zeros(len(classification_validation))
                current_matrix[(classification_validation == j) & (input_data[3] == i)] = 1
                confusion[i][j] = np.sum(current_matrix)

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        sns.heatmap(confusion,
                    annot=True,  # print pixel values onto each pixel
                    fmt=".0f",  # don't print decimals
                    cmap="gray",  # these are grayscale images, so "gray" is an appropriate colormap
                    ax=ax
                    # we have already initialised a matplotlib ax object we would like to use, so we provide it here
                    );
        ax.set_title("Validationdata: Confusion matrix")
        fig.show()
        sum = 0

        for i in range(8):
            print("Validationdata: Accuracy of " + str(i) + " is " + str(confusion[i][i] / np.sum(confusion[i])))
            sum += confusion[i][i]

        print("Validationdata: Overall accuracy is " + str(sum / np.sum(confusion)))

        print(f"Validationdata F1 score: {Metrics.f1_score_per_class(classification_validation, input_data[3])}")
        print(f"Validationdata F1 score: {Metrics.f1_score_average(classification_validation, input_data[3])}")

        return input_data
