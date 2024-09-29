import numpy as np
import pandas as pd
import os
import json
from collections import deque
from src.pipeline.interface_pipeline_component import PipelineComponent
from src.util.preprocess.feature_scaling import StandardScaler


class DynamicSamplesBuilder(PipelineComponent):
    def __init__(self):
        super().__init__()
        self.input_type = pd.DataFrame
        self.output_type = (list, np.ndarray)
        self.frame_queue = deque(maxlen=self.sample_size)

    def run(self, input_data):
        new_frame = input_data
        new_frame = new_frame.drop(columns = ["timestamp"])
        self.frame_queue.append(new_frame)
        if len(self.frame_queue) < self.sample_size:
            for _ in range(self.sample_size-1):
                self.frame_queue.append(new_frame)
        sample_df = pd.concat([df for df in self.frame_queue])


        return [sample_df], None

    def feature_extraction_mean_angle_diff(self, sample):

        # get mean angle change in left hand movement
        coordinates_left_hand = sample.loc[:, ["left_wrist_x", "left_wrist_y"]].to_numpy()
        vectors_of_movement_left = np.diff(coordinates_left_hand, axis=0)
        angle_diff_left = np.diff(np.arctan2(vectors_of_movement_left[:, 1], vectors_of_movement_left[:, 0]))
        angle_diff_left_abs = np.abs(angle_diff_left)
        angle_diff_left_abs[angle_diff_left_abs > np.pi] = 2 * np.pi - angle_diff_left_abs[angle_diff_left_abs > np.pi]
        angle_diff_average_left = angle_diff_left_abs.mean()

        # get mean angle change in right hand movement
        coordinates_right_hand = sample.loc[:, ["right_wrist_x", "right_wrist_y"]].to_numpy()
        vectors_of_movement_right = np.diff(coordinates_right_hand, axis=0)
        angle_diff_right = np.diff(np.arctan2(vectors_of_movement_right[:, 1], vectors_of_movement_right[:, 0]))
        angle_diff_right_abs = np.abs(angle_diff_right)
        angle_diff_right_abs[angle_diff_right_abs > np.pi] = 2 * np.pi - angle_diff_right_abs[angle_diff_right_abs > np.pi]
        angle_diff_average_right = angle_diff_right_abs.mean()

        # angle_diff_left_abs = np.concatenate(([0], angle_diff_left_abs, [0]))
        # angle_diff_right_abs = np.concatenate(([0], angle_diff_right_abs, [0]))

        sample["left_hand_angle_diff"] = angle_diff_average_left
        sample["right_hand_angle_diff"] = angle_diff_average_right

        return sample

    def feature_extraction_delta(self, sample: pd.DataFrame) -> pd.DataFrame:

        coordinates_left_hand_x = sample.loc[:, ["left_wrist_x"]].to_numpy().flatten()
        vectors_of_movement_left_x = np.diff(coordinates_left_hand_x, axis=0)
        vectors_of_movement_left_x = np.concatenate(([0], vectors_of_movement_left_x))

        coordinates_right_hand_x = sample.loc[:, ["right_wrist_x"]].to_numpy().flatten()
        vectors_of_movement_right_x = np.diff(coordinates_right_hand_x, axis=0)
        vectors_of_movement_right_x = np.concatenate(([0], vectors_of_movement_right_x))

        coordinates_left_hand_y = sample.loc[:, ["left_wrist_y"]].to_numpy().flatten()
        vectors_of_movement_left_y = np.diff(coordinates_left_hand_y, axis=0)
        vectors_of_movement_left_y = np.concatenate(([0], vectors_of_movement_left_y))

        coordinates_right_hand_y = sample.loc[:, ["right_wrist_y"]].to_numpy().flatten()
        vectors_of_movement_right_y = np.diff(coordinates_right_hand_y, axis=0)
        vectors_of_movement_right_y = np.concatenate(([0], vectors_of_movement_right_y))

        sample["left_wrist_delta_x"] = vectors_of_movement_left_x
        sample["right_wrist_delta_x"] = vectors_of_movement_right_x
        sample["left_wrist_delta_y"] = vectors_of_movement_left_y
        sample["right_wrist_delta_y"] = vectors_of_movement_right_y

        return sample

    def standardize_with_loaded_scaler(self, input_data):
        scaler = StandardScaler()

        with open(os.path.join(os.getcwd(), f'resource/models/{self.model_name}_scaler.json'),
                  "r") as f:
            scaler_data = json.load(f)

        scaler.mean = np.array(scaler_data["mean"])
        scaler.std = np.array(scaler_data["std"])

        return scaler.transform(input_data)


if __name__ == "__main__":
    frame1 = pd.DataFrame({'A': [3], 'B': [4], 'C': [5]})
    frame2 = pd.DataFrame({'A': [3], 'B': [5], 'C': [1]})
    frame3 = pd.DataFrame({'A': [6], 'B': [5], 'C': [7]})
    dynamic_samples_builder = DynamicSamplesBuilder()
    print(dynamic_samples_builder.run(frame1))
    print(dynamic_samples_builder.run(frame2))
    print(dynamic_samples_builder.run(frame3))
    print(type(dynamic_samples_builder.run(frame3)))


