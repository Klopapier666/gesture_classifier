import numpy as np
import pandas as pd
from src.pipeline.interface_pipeline_component import PipelineComponent


class Classifier(PipelineComponent):
    def __init__(self, nn_model):
        super().__init__()
        # only the first array interests us, just for reasons of compatability the other arrays exist
        self.input_type = (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        self.output_type = str
        self.nn_model = nn_model

    def run(self, input_data):
        output_data = self.nn_model.classification(input_data[0])
        output_data_str = self.mapping(output_data)
        return output_data_str

    def mapping(self, input_data):
        input_data = [k for k, v in self.label_map.items() if v == input_data[0]]
        return input_data[0]