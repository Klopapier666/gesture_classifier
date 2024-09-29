#from src.pipeline.pipeline_components.data_loader import DataLoader
#from src.pipeline.pipeline_components.preprocessor import Preprocessor
#from src.pipeline.pipeline_components.samples_builder import SamplesBuilder
#from src.pipeline.pipeline_components.model_trainer import ModelTrainer
from src.pipeline.pipeline import Pipeline
#from src.machine_learning_framework.TrainingModel import *
#
## create model
#layer_1 = ClassicLayer(421, 20)
#layer_2 = ClassicLayer(20, 20)
#layer_3 = ClassicLayer(20, 20)
#layer_4 = SoftmaxLayer(20, 4)
#
#
#model = NeuronalNetworkModel([layer_1, layer_2, layer_3, layer_4])
#
#data_loader = DataLoader()
#preprocessor = Preprocessor(training=True)
#sample_builder = SamplesBuilder()
#model_trainer = ModelTrainer(model, 8, 0.5, 1)


pipeline = Pipeline(mode="train")

output_data = pipeline.run()



