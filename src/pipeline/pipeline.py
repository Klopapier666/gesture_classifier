from src.pipeline.pipeline_components.classifier import Classifier
from src.pipeline.pipeline_components.data_loader import DataLoader
from src.pipeline.pipeline_components.dynamic_samples_builder import DynamicSamplesBuilder
from src.pipeline.pipeline_components.gesture_fire_controller import GestureFireController
from src.pipeline.pipeline_components.model_trainer import ModelTrainer
from src.pipeline.pipeline_components.preprocessor import Preprocessor
from src.pipeline.pipeline_components.samples_builder import SamplesBuilder
from src.pipeline.pipeline_components.preprocessor2 import Preprocessor2
from src.machine_learning_framework.TrainingModel import *
from src.pipeline.pipeline_components.hyperparameter_tester import Hyperparameter_Tester
from src.pipeline.pipeline_components.saved_samples_loader import SavedSamplesLoader


class Pipeline:
    #def __init__(self, *components):
    #    self.components = []
    #    self._add_components(components)

    def __init__(self, mode):
        self.components = []

        if mode == "live_mandatory":
            #with open(os.path.join(os.getcwd(), 'resource/models/final_model_mandatory/saved_weights_final_model_mandatory.pkl'), 'rb') as file:
            with open(os.path.join(os.getcwd(),
                                   'resource/models/model_50_20_4/gesicherte_gewichte_Model_50_20_4.pkl'),'rb') as file:
                loaded_list = pickle.load(file)

            loaded_weights, loaded_bias = loaded_list
            # Create model
            layer_1 = ClassicLayer(899, 50, loaded_weights[0], loaded_bias[0])
            layer_2 = ClassicLayer(50, 20, loaded_weights[1], loaded_bias[1])
            layer_3 = SoftmaxLayer(20, 4, loaded_weights[2], loaded_bias[2])
            model = NeuronalNetworkModel([layer_1, layer_2, layer_3])

            #abs_path_scaler = os.path.join(os.getcwd(), f'resource/models/final_model_mandatory/final_model_mandatory_scaler.json')
            abs_path_scaler = os.path.join(os.getcwd(),
                                           f'resource/models/model_50_20_4/Model_50_20_4_scaler.json')

            preprocessor = Preprocessor(False)
            dynamic_samples_builder = DynamicSamplesBuilder()
            preprocessor2 = Preprocessor2(abs_path_scaler=abs_path_scaler, training=False)
            classifier = Classifier(model)
            gesture_fire_controller = GestureFireController()
            components = [preprocessor, dynamic_samples_builder, preprocessor2, classifier, gesture_fire_controller]
            self._add_components(components)

        if mode == "live_all":
            with open(os.path.join(os.getcwd(), 'resource/models/final_model_all/saved_weights_final_model_all.pkl'), 'rb') as file:
                loaded_list = pickle.load(file)

            loaded_weights, loaded_bias = loaded_list
            # Create model
            layer_1 = ClassicLayer(899, 100, loaded_weights[0], loaded_bias[0])
            layer_2 = ClassicLayer(100, 100, loaded_weights[1], loaded_bias[1])
            layer_3 = SoftmaxLayer(100, 8, loaded_weights[2], loaded_bias[2])
            model = NeuronalNetworkModel([layer_1, layer_2, layer_3])

            abs_path_scaler = os.path.join(os.getcwd(), f'resource/models/final_model_all/final_model_all_scaler.json')

            preprocessor = Preprocessor(False)
            dynamic_samples_builder = DynamicSamplesBuilder()
            preprocessor2 = Preprocessor2(abs_path_scaler=abs_path_scaler, training=False)
            classifier = Classifier(model)
            gesture_fire_controller = GestureFireController()
            components = [preprocessor, dynamic_samples_builder, preprocessor2, classifier, gesture_fire_controller]
            self._add_components(components)

        if mode == "train":
            # Create model
            layer_1 = ClassicLayer(899, 100)
            layer_2 = ClassicLayer(100, 100)
            layer_3 = SoftmaxLayer(100, 8)
            model = NeuronalNetworkModel([layer_1, layer_2, layer_3])
            saved_samples_loader = SavedSamplesLoader()

            model_trainer = ModelTrainer(model, 350, 0.3, 0.1)
            #model_trainer = ModelTrainer(model, 1, 0.5, 0.2)
            components = [saved_samples_loader, model_trainer]
            self._add_components(components)

        if mode == "hyperparameter":
            saved_samples_loader = SavedSamplesLoader()
            hyperparameter_tester = Hyperparameter_Tester()
            components = [saved_samples_loader, hyperparameter_tester]
            self._add_components(components)

        if mode == "without_preprocessing":
            layer_1 = ClassicLayer(4110, 50)
            layer_2 = ClassicLayer(50, 20)
            layer_3 = SoftmaxLayer(20, 8)
            model = NeuronalNetworkModel([layer_1, layer_2, layer_3])

            data_loader = DataLoader()
            preprocessor = Preprocessor(training=True, without_preprocessing=True)
            sample_builder = SamplesBuilder(without_preprocessing=True)
            preprocessor2 = Preprocessor2(training=True, without_preprocessing2=True)
            model_trainer = ModelTrainer(model, 300, 0.3, 0.5)
            components = [data_loader, preprocessor, sample_builder, preprocessor2, model_trainer]
            self._add_components(components)

        if mode == "sample_building":
            abs_path = os.path.join(os.getcwd(), 'test.json')

            data_loader = DataLoader()
            preprocessor = Preprocessor(training=True)
            sample_builder = SamplesBuilder()
            preprocessor2 = Preprocessor2(abs_path_scaler= abs_path, training=True)
            components = [data_loader, preprocessor, sample_builder, preprocessor2]
            self._add_components(components)


    def run(self, input_data=None):
        output_data = input_data  # Init output_data to input_data for first component
        for component in self.components:
            output_data = component.run(output_data)
        return output_data

    def _add_components(self, components):
        if not components:
            raise TypeError("No components provided.")

        for component in components:
            if self._validate_component(component):
                self.components.append(component)
            else:
                raise TypeError("Output types of components do not match.")

    def _validate_component(self, component):
        if not self.components:
            return True
        last_output_type = self.components[-1].output_type
        return component.input_type == last_output_type
