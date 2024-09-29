import os
import pandas as pd
from src.pipeline.interface_pipeline_component import PipelineComponent


class DataLoader(PipelineComponent):
    def __init__(self):
        self.input_type = None
        self.output_type = pd.DataFrame
        self.directory = os.path.join(os.getcwd(), 'resource/training_data_all')

    def run(self, input_data=None) -> pd.DataFrame:
        print(f"Running component {self.__class__.__name__}...")
        csv_files = [file for file in os.listdir(self.directory) if file.endswith('.csv')]

        dfs = []
        for file in csv_files:
            print(file)
            file_path = os.path.join(self.directory, file)
            df = pd.read_csv(file_path)
            dfs.append(df)
        output = pd.concat(dfs, ignore_index=True)
        print(output.shape)
        return output

