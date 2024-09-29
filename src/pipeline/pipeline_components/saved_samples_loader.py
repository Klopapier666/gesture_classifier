import os
import json

import numpy as np

from src.pipeline.interface_pipeline_component import PipelineComponent


class SavedSamplesLoader(PipelineComponent):
    def __init__(self):
        self.input_type = None
        self.output_type = (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        self.file = os.path.join(os.getcwd(), 'resource/samples/final_sampled_data_all.npz')
        #self.file = os.path.join(os.getcwd(), 'resource/models/final_model_mandatory/data.npz')

    def run(self, input_data=None):
        loaded_data = np.load(self.file)
        X_train = loaded_data['array1']
        y_train = loaded_data['array2']
        X_validation = loaded_data['array3']
        y_validation = loaded_data['array4']
        X_test = loaded_data['array5']
        y_test = loaded_data['array6']
        return X_train, y_train, X_validation, y_validation, X_test, y_test

