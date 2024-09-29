import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import os
from src.pipeline.interface_pipeline_component import PipelineComponent
from src.util.preprocess.feature_scaling import StandardScaler


class Preprocessor2(PipelineComponent):
    def __init__(self, abs_path_scaler = "" ,training=False, without_preprocessing2=False):
        super().__init__()
        self.input_type = (list, np.ndarray)
        self.output_type = (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        self.training = training
        self.abs_path_scaler = abs_path_scaler

        self.without_preprocessing2 = without_preprocessing2

    def run(self, input_data: (list, np.ndarray)) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        X = input_data[0]
        y = input_data[1]

        # just a test for the final presentation, how the model performes without preprocessing
        if(self.without_preprocessing2):
            X_samples = np.zeros((len(X), (self.sample_size * (X[0].shape[1] + 4))))
            for index, sample in tqdm(enumerate(X_samples)):
                X_samples[index] = sample.reshape(-1)
            X_train, y_train, X_validation, y_validation, X_test, y_test = self._train_validation_test_split(X_samples, y)
            return X_train, y_train, X_validation, y_validation, X_test, y_test

        X = self.feature_extraction(X)

        if(self.training):
            X_train, y_train, X_validation, y_validation, X_test, y_test = self._train_validation_test_split(X, y)
            print(f"X_train:        Shape: {str(X_train.shape):<15} Type: {str(type(X_train))}")
            print(f"y_train:        Shape: {str(y_train.shape):<15} Type: {str(type(y_train))}")
            print(f"X_validation:   Shape: {str(X_validation.shape):<15} Type: {str(type(X_validation))}")
            print(f"y_validation:   Shape: {str(y_validation.shape):<15} Type: {str(type(y_validation))}")
            print(f"X_test:         Shape: {str(X_test.shape):<15} Type: {str(type(X_test))}")
            print(f"y_test:         Shape: {str(y_test.shape):<15} Type: {str(type(y_test))}")

            X_train = self.standardize_and_save_to_json(X_train)
            X_validation = self.standardize_with_loaded_scaler(X_validation)
            X_test = self.standardize_with_loaded_scaler(X_test)


            X_train = self.add_bias(X_train)
            X_validation = self.add_bias(X_validation)
            X_test = self.add_bias(X_test)

            self.save_samples(X_train, y_train, X_validation, y_validation, X_test, y_test)

        else:
            # in this case, X_train is our sample (nothing to do with training
            X_train = X
            y_train = X_validation = y_validation = X_test = y_test =None
            #if(self.training):
                #X_train = self.standardize_and_save_to_json(X_train)
            #else:
            X_train = self.standardize_with_loaded_scaler(X_train)
            X_train = self.add_bias(X_train)

            #with open(os.path.join(os.getcwd(), f'resource/testing/sample_dynamic_testing.csv'), 'a') as file:
                #np.savetxt(file, X_train, delimiter=",")

        return X_train, y_train, X_validation, y_validation, X_test, y_test


    @staticmethod
    def _train_validation_test_split(X, y):
        number_of_samples = X.shape[0]
        train_size = int(0.7 * number_of_samples)
        validation_size = int(0.15 * number_of_samples)

        indices = np.arange(number_of_samples)
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        validation_indices = indices[train_size:train_size + validation_size]
        test_indices = indices[train_size + validation_size:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_validation, y_validation = X[validation_indices], y[validation_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        return X_train, y_train, X_validation, y_validation, X_test, y_test

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

        #angle_diff_left_abs = np.concatenate(([0], angle_diff_left_abs, [0]))
        #angle_diff_right_abs = np.concatenate(([0], angle_diff_right_abs, [0]))

        #sample["left_hand_angle_diff"] = angle_diff_average_left
        #sample["right_hand_angle_diff"] = angle_diff_average_right

        return np.array([angle_diff_average_left, angle_diff_average_right])

    def feature_extraction_delta(self, sample: pd.DataFrame) -> pd.DataFrame:

        coordinates_left_hand_x = sample.loc[:, ["left_wrist_x"]].to_numpy().flatten()
        vectors_of_movement_left_x = np.diff(coordinates_left_hand_x, axis=0)

        coordinates_right_hand_x = sample.loc[:, ["right_wrist_x"]].to_numpy().flatten()
        vectors_of_movement_right_x = np.diff(coordinates_right_hand_x, axis=0)

        coordinates_left_hand_y = sample.loc[:, ["left_wrist_y"]].to_numpy().flatten()
        vectors_of_movement_left_y = np.diff(coordinates_left_hand_y, axis=0)

        coordinates_right_hand_y = sample.loc[:, ["right_wrist_y"]].to_numpy().flatten()
        vectors_of_movement_right_y = np.diff(coordinates_right_hand_y, axis=0)


        vectors_of_movement = np.concatenate([vectors_of_movement_left_x, vectors_of_movement_right_x, vectors_of_movement_left_y, vectors_of_movement_right_y])

        return vectors_of_movement

    def standardize_and_save_to_json(self, input_data):
        scaler = StandardScaler()
        scaler.fit(input_data)
        # save mean and std to json
        json_dict = {
            "model_name": self.model_name,
            "mean": scaler.mean.tolist(),
            "std": scaler.std.tolist()
        }
        with open(self.abs_path_scaler, "x") as f:
            json.dump(json_dict, f)

        return scaler.transform(input_data)

    def standardize_with_loaded_scaler(self, input_data):
        scaler = StandardScaler()

        #with open(os.path.join(os.getcwd(), f'resource/models/{self.model_name}_scaler.json'), "r") as f:
        with open(self.abs_path_scaler, "r") as f:
            scaler_data = json.load(f)

        scaler.mean = np.array(scaler_data["mean"])
        scaler.std = np.array(scaler_data["std"])

        return scaler.transform(input_data)


    def save_samples(self, X_train, y_train, X_validation, y_validation, X_test, y_test):
        np.savez(os.path.join(os.getcwd(), f'resource/models/final_model_mandatory/data.npz'), array1=X_train, array2=y_train, array3=X_validation, array4=y_validation, array5=X_test, array6=y_test)

    def add_bias(self, X_samples):

        X_samples = X_samples.reshape(X_samples.shape[0], -1)
        one_matrix = np.ones((X_samples.shape[0], 1))
        X_samples = np.column_stack((one_matrix, X_samples))

        return X_samples

    def feature_extraction(self, X_samples):
        test = X_samples[0].shape[1]
        X = np.zeros((len(X_samples), (self.sample_size * (X_samples[0].shape[1] + 4) + 2 - 4)))
        for index, sample in tqdm(enumerate(X_samples)):
            vector_of_movement = self.feature_extraction_delta(sample)
            mean_angle = self.feature_extraction_mean_angle_diff(sample)
            sample_np = np.concatenate((sample.values.reshape(-1), vector_of_movement, mean_angle))
            X[index] = sample_np
        return X