import pandas as pd
import numpy as np
import json
import os

from src.pipeline.interface_pipeline_component import PipelineComponent
from src.util.preprocess.feature_scaling import StandardScaler


class Preprocessor(PipelineComponent):
    def __init__(self, training=False, without_preprocessing=False):
        super().__init__()
        self.input_type = pd.DataFrame
        self.output_type = pd.DataFrame
        self.training = training
        self.without_preprocessing = without_preprocessing

    def run(self, input_data):
        """
        Run method
        """
        # just for the final presentation to show what happens without preprocessing
        if (self.without_preprocessing):
             return self.encode_labels(input_data)

        #print(f"Running component {self.__class__.__name__}...")
        if self.training:
            input_data = input_data.loc[:, self.feature_list_with_truth]
            input_data = self.encode_labels(input_data)
        else:
            input_data = input_data.loc[:, self.feature_list_without_truth]


        input_data = self.center_body(input_data)
        input_data = self.scale_body(input_data)

        # exclude hip
        input_data = input_data.drop(columns=['left_hip_x', 'left_hip_y', 'right_hip_x', 'right_hip_y'])

        #if self.training:
         #   input_data = self.standardize_and_save_to_json(input_data)
        #else:
         #   input_data = self.standardize_with_loaded_scaler(input_data)

        # input_data.to_csv(os.path.join(os.getcwd(), f'../resource/testing/preprocessed_data.csv'), index=False) # for testing
        #print("Preprocessing done!")
        #print("Shape after preprocessing: ", input_data.shape)
        return input_data

    @staticmethod
    def middle_point(x1, x2):
        return (x1 + x2) / 2

    def center_body(self, input_data):
        middle_of_shoulders_x = self.middle_point(input_data.loc[:, 'left_shoulder_x'],
                                                  input_data.loc[:, 'right_shoulder_x'])
        middle_of_shoulders_y = self.middle_point(input_data.loc[:, 'left_shoulder_y'],
                                                  input_data.loc[:, 'right_shoulder_y'])
        # middle_of_shoulders_z = self.middle_point(input_data.loc[:, 'left_shoulder_z'],
        #                                           input_data.loc[:, 'right_shoulder_z'])

        # Subtract middle_of_shoulder coordinates from all other coordinates
        for column in input_data.columns:
            if not(column.startswith('ground_truth') or column.startswith('time')):
                if column.endswith('x'):
                    input_data.loc[:, column] -= middle_of_shoulders_x
                elif column.endswith('y'):
                    input_data.loc[:, column] -= middle_of_shoulders_y
                # elif column.endswith('z'):
                #     input_data.loc[:, column] -= middle_of_shoulders_z

        return input_data

    def scale_body(self, input_data):
        middle_of_shoulders_x = self.middle_point(input_data.loc[:, 'left_shoulder_x'],
                                                  input_data.loc[:, 'right_shoulder_x'])
        middle_of_shoulders_y = self.middle_point(input_data.loc[:, 'left_shoulder_y'],
                                                  input_data.loc[:, 'right_shoulder_y'])
        # middle_of_shoulders_z = self.middle_point(input_data.loc[:, 'left_shoulder_z'],
        #                                           input_data.loc[:, 'right_shoulder_z'])

        middle_of_hips_x = self.middle_point(input_data.loc[:, 'left_hip_x'], input_data.loc[:, 'right_hip_x'])
        middle_of_hips_y = self.middle_point(input_data.loc[:, 'left_hip_y'], input_data.loc[:, 'right_hip_y'])
        # middle_of_hips_z = self.middle_point(input_data.loc[:, 'left_hip_z'], input_data.loc[:, 'right_hip_z'])

        TARGET_HEIGHT_HIP_TO_SHOULDER = 0.8  # in meters

        # euclidean distance between hips and shoulders
        hip_shoulder_distance = np.sqrt(
            (middle_of_hips_x - middle_of_shoulders_x) ** 2 +
            (middle_of_hips_y - middle_of_shoulders_y) ** 2
            # + (middle_of_hips_z - middle_of_shoulders_z) ** 2
        )

        scale_factor = TARGET_HEIGHT_HIP_TO_SHOULDER / hip_shoulder_distance

        for column in input_data.columns:
            if not (column.startswith('ground_truth') or column.startswith('time')):
                input_data.loc[:, column] *= scale_factor

        return input_data

    def standardize_and_save_to_json(self, input_data):
        scaler = StandardScaler()
        scaler.fit(input_data)
        # save mean and std to json
        json_dict = {
            "model_name": self.model_name,
            "mean": scaler.mean.to_dict(),
            "std": scaler.std.to_dict()
        }
        with open(os.path.join(os.getcwd(), f'resource/models/{self.model_name}_scaler.json'), "w") as f:
            json.dump(json_dict, f)

        return scaler.transform(input_data)

    def standardize_with_loaded_scaler(self, input_data):
        scaler = StandardScaler()

        with open(os.path.join(os.getcwd(), f'resource/models/{self.model_name}_scaler.json'), "r") as f:
            scaler_data = json.load(f)

        scaler.mean = scaler_data["mean"]
        scaler.std = scaler_data["std"]

        return scaler.transform(input_data)

    def encode_labels(self, input_data):
        input_data['ground_truth'] = input_data['ground_truth'].map(self.label_map)
        return input_data
