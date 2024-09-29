import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import os
from src.pipeline.interface_pipeline_component import PipelineComponent
from src.util.preprocess.feature_scaling import StandardScaler


class SamplesBuilder(PipelineComponent):
    def __init__(self, without_preprocessing=False):
        super().__init__()
        self.input_type = pd.DataFrame
        self.output_type = (list, np.ndarray)
        self.without_preprocessing = without_preprocessing

    def run(self, input_data: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        print(f"Running component {self.__class__.__name__}...")
        X = input_data.drop(columns=["ground_truth"])
        y = input_data["ground_truth"]

        test = input_data[input_data["ground_truth"] != 0]
        test = test[test["ground_truth"] != 1]
        test = test[test["ground_truth"] != 2]
        test = test[test["ground_truth"] != 3]
        print(test)


        #y = self._encode_labels(y)  # move to preprocessing
        y = y.rename('encoded_labels')
        print(f"X_samples:      Shape: {str(X.shape):<15} Type: {str(type(X))}")
        print(f"y_samples:      Shape: {str(y.shape):<15} Type: {str(type(y))}")

        X, y = self._build_samples(X, y)
        print(f"X_samples:      Shape: {str(len(X)):<15} Type: {str(type(X))}, Type samples: {str(type(X[0]))}")
        print(f"y_samples:      Shape: {str(y.shape):<15} Type: {str(type(y))}")

        if(not self.without_preprocessing):
            X, y = self._remove_idles(X, y)

        print(f"Idle-Counter: {np.sum(y == 0)}")
        print(f"SwipeLeft-Counter: {np.sum(y == 1)}")
        print(f"SwipeRight-Counter: {np.sum(y == 2)}")
        print(f"Rotate-RightCounter: {np.sum(y == 3)}")
        print(f"Rotate-Left-Counter: {np.sum(y == 4)}")
        print(f"Spread-Counter: {np.sum(y == 5)}")
        print(f"Pinch-Counter: {np.sum(y == 6)}")
        print(f"Flip-Table-Counter: {np.sum(y == 7)}")

        return X, y

    def _build_samples(self, X, y):
        number_of_frames = len(X)
        X_samples = [None]*(number_of_frames - self.sample_size + 1)
        y_samples = np.zeros((number_of_frames - self.sample_size + 1))

        samples_counter = 0
        dropped_samples_counter = 0
        for i in tqdm(range(number_of_frames - self.sample_size + 1)):
            current_frame_index = i + self.sample_size - 1
            X_sample = X.iloc[current_frame_index - self.sample_size + 1: current_frame_index + 1]
            y_sample = y.iloc[current_frame_index - self.sample_size + 1: current_frame_index + 1]
            if self._check_sample(X_sample):
                X_sample = X_sample.drop(columns=["timestamp"])
                X_samples[samples_counter] = X_sample
                y_samples[samples_counter] = y_sample.value_counts().idxmax()
                samples_counter += 1
            else:
                dropped_samples_counter += 1

        print(f"Dropped samples: {dropped_samples_counter:>8}")

        X_samples = X_samples[:samples_counter]
        y_samples = y_samples[:samples_counter]

        #X_samples = X_samples.reshape(X_samples.shape[0], -1)

        return X_samples, y_samples

    @staticmethod
    def _check_sample(X_sample):
        if not X_sample["timestamp"].is_monotonic_increasing:  # Check if sample is on edge of video (csv-file)
            return False
        return True

    @staticmethod
    #def _remove_idles(X, y):
    #    number_of_samples = X.shape[0]
    #    indices = np.arange(number_of_samples)
    #    np.random.shuffle(indices)
    #    number_of_idle = np.sum(y == 0)
    #    number_of_others = np.sum(y != 0)
    #    indices_to_delete = []
    #    for index in tqdm(indices):
    #        if(number_of_idle/(number_of_idle + number_of_others) > 1/3) and index < y.shape[0] and y[index] == 0:
    #            number_of_idle -= 1
    #            indices_to_delete.append(index)
    #            if(number_of_idle/(number_of_idle + number_of_others) <= 1/3):
    #                X = np.delete(X, indices_to_delete, axis=0)
    #                y = np.delete(y, indices_to_delete)
    #                return X, y

    #    X = np.delete(X, indices_to_delete, axis=0)
    #    y = np.delete(y, indices_to_delete)

    #    return X, y

    def _remove_idles(X, y):
        number_of_samples = len(X)
        indices = np.arange(number_of_samples)
        np.random.shuffle(indices)
        number_of_idle = np.sum(y == 0)
        number_of_others = np.sum(y != 0)
        indices_to_delete = []
        for index in indices:
            if(number_of_idle/(number_of_idle + number_of_others) > 1/3) and index < y.shape[0] and y[index] == 0:
                number_of_idle -= 1
                indices_to_delete.append(index)
                if number_of_idle/(number_of_idle + number_of_others) <= 1/3:
                    for index in sorted(indices_to_delete, reverse=True):
                        del X[index]
                    y = np.delete(y, indices_to_delete)
                    return X, y

        X = [sample for idx, sample in enumerate(X) if idx not in indices_to_delete]
        y = np.delete(y, indices_to_delete)

        return X, y

    @staticmethod
    def _encode_labels(y):
        label_map = {label: idx for idx, label in enumerate(pd.unique(y))}
        print(label_map)

        encoded_labels = [label_map[label] for label in y]

        return pd.DataFrame(encoded_labels, columns=["encoded_labels"])



    def test(self):
        pass
        # y.value_counts().get('rotate', 0)
        # print(f"gestures before encoding {y.value_counts().get('rotate', 0)}")
        # print(f"gestures after encoding: {np.count_nonzero(y == 1)}")
