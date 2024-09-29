import numpy as np
from src.pipeline.interface_pipeline_component import PipelineComponent
from src.machine_learning_framework.TrainingModel import *
from src.machine_learning_framework import HyperparameterTesting
import seaborn as sns

class Hyperparameter_Tester(PipelineComponent):
    def __init__(self):
        super().__init__()
        self.input_type = (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        self.output_type = None


    def run(self, input_data):

        layer = [3, 4]
        neuron = [40, 80, 100]
        epoch = [250]
        #epoch = [3]
        learning_rate = [0.3, 0.5]
        lam = [0.1, 0.5, 1, 2]
        abs_path = os.path.join(os.getcwd(), 'resource/testing/hyperparameter_test2/')

        HyperparameterTesting.randomized_search(input_data[0], input_data[1], input_data[2], input_data[3], layer, neuron, learning_rate, epoch, lam, 15, 8, abs_path)


