import pandas as pd
import os


class LoadICARuns(object):
    def __init__(self, directory, sample_path):
        # Directory were the smaller splits are
        self.directory = directory
        # File were the sample dataset is located
        self.sample_path = sample_path
        self.sample_data = None
        # Dictionary to save the small splits
        self.small_dataframes = {}
        # Load the datasets
        self.load_small()
        self.load_sample()
        # Merge small datasets
        self.combined_small = None
        self.merge_small()
        # Small and sample
        self.df_full = None

    def load_small(self):
        # Loop over the directory
        for entry in os.scandir(self.directory):
            # Load the information of the separate splits these clusters should be called 'ICARUN_SPLIT'
            if 'ICARUN_SPLIT' in entry.path:
                file_path = f'{entry.path}/ica_independent_components_consensus.tsv'
                # Get the split name should be the last part of the folder
                name = entry.path.split('_')[-1]
                # Load the component
                df = pd.read_csv(file_path, sep='\t', index_col=0)
                # Print it to check
                print(f"Number of components split {name}: {df.shape[1]}")
                # Set the names as the split type
                df.columns = [f'{x}_{name}' for x in df.columns]
                # Add id and dataframe to dictionary
                self.small_dataframes[name] = df

    def load_sample(self):
        # Load the sample dataset
        self.sample_data = pd.read_csv(self.sample_path, sep='\t', index_col=0)
        print(f'Number of components All data: {self.sample_data.shape[1]}')
        # Make the columns unique so they later can be added to the same dataframe
        self.sample_data.columns = [f'{x}_big' for x in self.sample_data.columns]

    def merge_small(self):
        # Merge the small components
        first_run = True
        for name in self.small_dataframes:
            if first_run:
                self.combined_small = self.small_dataframes[name].copy()
                first_run = False
            else:
                self.combined_small = self.combined_small.join(self.small_dataframes[name])

    def sample_merge_small(self):
        self.df_full = self.combined_small.join(self.sample_data)

    def get_sample_data(self):
        return self.sample_data

    def get_individual_small(self):
        return self.small_dataframes

    def get_merged_small(self):
        return self.combined_small

    def get_all(self):
        return self.df_full
