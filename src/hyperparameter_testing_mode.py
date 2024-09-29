from src.pipeline.pipeline_components.data_loader import DataLoader
from src.pipeline.pipeline_components.preprocessor import Preprocessor
from src.pipeline.pipeline_components.samples_builder import SamplesBuilder
from src.pipeline.pipeline_components.hyperparameter_tester import Hyperparameter_Tester
from src.pipeline.pipeline import Pipeline
from src.machine_learning_framework.TrainingModel import *




pipeline = Pipeline("hyperparameter")

output_data = pipeline.run()



