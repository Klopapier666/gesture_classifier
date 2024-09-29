from abc import ABC, abstractmethod


class PipelineComponent(ABC):
    @abstractmethod
    def __init__(self):
        self.input_type = None
        self.output_type = None
        self.model_name = ("final_model_all"
                           "")
        self.sample_size = 30
        self.feature_list_without_truth = [
            "timestamp",
            "nose_x", "nose_y",
            "left_shoulder_x", "left_shoulder_y",
            "right_shoulder_x", "right_shoulder_y",
            "left_elbow_x", "left_elbow_y",
            "right_elbow_x", "right_elbow_y",
            "left_wrist_x", "left_wrist_y",
            "right_wrist_x", "right_wrist_y",
            "left_hip_x", "left_hip_y",
            "right_hip_x", "right_hip_y",
            "left_pinky_x", "left_pinky_y",
            "right_pinky_x", "right_pinky_y",
            "left_index_x", "left_index_y",
            "right_index_x", "right_index_y",
            "left_thumb_x", "left_thumb_y",
            "right_thumb_x", "right_thumb_y"
            ]
        self.label_map = {
            "idle": 0,
            "swipe_left": 1,
            "swipe_right": 2,
            "rotate_right": 3,
            "rotate_left": 4,
            "spread": 5,
            "pinch": 6,
            "flip_table": 7
        }
        self.feature_list_with_truth = self.feature_list_without_truth.copy()
        self.feature_list_with_truth.append("ground_truth")

    @abstractmethod
    def run(self, input_data):
        """
        Method to run the component on input data.
        """
        pass
