import pandas as pd
import os


class LoadICARuns(object):
    """
    Class to load all the different ESes datasets
    """
    def __init__(self, directory, sample_path):
        """
        input variables:
            directory: Directory where the folders to the subset ESes exist
            sample_path: Path to the sample ESes
        """
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
        self.df_full = self.combined_small.join(self.sample_data)
        # Credibility index
        self.credibility_small = None
        self.credibility_big = None
        self.credibility_sets = None
        self.load_credibility()

    def load_small(self):
        """
        Load all the subset ESes
        """
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
        """
          Load all the sample ESes
        """
        # Load the sample dataset
        self.sample_data = pd.read_csv(self.sample_path, sep='\t', index_col=0)
        print(f'Number of components All data: {self.sample_data.shape[1]}')
        # Make the columns unique so they later can be added to the same dataframe
        self.sample_data.columns = [f'{x}_big' for x in self.sample_data.columns]

    def merge_small(self):
        """
          Merge all the subsets ESes to one dataframe
        """
        # Merge the small components
        first_run = True
        for name in self.small_dataframes:
            if first_run:
                self.combined_small = self.small_dataframes[name].copy()
                first_run = False
            else:
                self.combined_small = self.combined_small.join(self.small_dataframes[name])

    def get_sample_data(self):
        return self.sample_data

    def get_individual_small(self):
        return self.small_dataframes

    def get_merged_small(self):
        return self.combined_small

    def get_all(self):
        return self.df_full

    def load_credibility(self):
        """
          Load the different credibility scores
        """
        self.credibility_small = []
        self.credibility_sets = []
        for entry in os.scandir(self.directory):
            if 'ICARUN_SPLIT' in entry.path:
                for file in os.scandir(entry):
                    if 'ica_robustness_metrics_independent_components_consensus.tsv' in file.path:
                        z = file.path.split('/')[-2][-1]
                        df = pd.read_csv(file.path, sep='\t', index_col=0)
                        df.index = [f'{q}_{z}' for q in df.index]
                        self.credibility_small.append(df)
                        self.credibility_sets.append(f'{z}')
        # Change index
        all_data = pd.read_csv(
            '/home/MarkF/DivideConquer/Results/2000_Samples_Experiment/Clustered_vs_Random_Experiment/ICARUN_ALL/'
            'ica_robustness_metrics_independent_components_consensus.tsv', sep='\t', index_col=0)
        all_data.index = [f'{z}_big' for z in all_data.index]
        self.credibility_sets.append(f'big')
        self.credibility_small.append(all_data)
        self.credibility_big = pd.concat(self.credibility_small)

    def get_credibility(self):
        return self.credibility_small, self.credibility_big, self.credibility_sets


class LoadCancer(LoadICARuns):
    """
    Class to load all the different ESes for the GPL570 dataset
    """
    def load_small(self):
        """
        Load the subsets ES
        """
        # Loop over the cancer type folder
        for cancer_type in os.scandir(self.directory):
            # For this specific case do not load all cancer data
            if 'All_Cancer' not in cancer_type.path:
                # Also do not load any files that may be present
                if os.path.isdir(cancer_type):
                    # TODO this doesnt have to be in a for loop
                    # Search th consensus file
                    name = cancer_type.path.split('/')[-1].replace('_', ' ')
                    # Load the file
                    df = pd.read_csv(f'{cancer_type.path}/ICARUN/ica_independent_components_consensus.tsv',
                                     sep='\t', index_col=0)
                    # Print for the check
                    print(f"Number of components split {name}: {df.shape[1]}")
                    # Give the column a unique name and add the relative values to the correct list
                    df.columns = [f'{x}_{name}' for x in df.columns]
                    self.small_dataframes[name] = df

    def load_credibility(self):
        """
        Load the credibility score
        """
        self.credibility_small = []
        self.credibility_sets = []
        for cancer_type in os.scandir(self.directory):
            if 'All_Cancer' not in cancer_type.path and os.path.isdir(cancer_type.path):
                for file in os.scandir(f'{cancer_type.path}/ICARUN'):
                    if 'ica_robustness_metrics_independent_components_consensus.tsv' in file.path:
                        i = cancer_type.path.split('/')[-1].replace('_', ' ')
                        df = pd.read_csv(file.path, sep='\t', index_col=0)
                        df.index = [f'{x}_{i}' for x in df.index]
                        self.credibility_small.append(df)
                        self.credibility_sets.append(i)
        # Change index
        all_data = pd.read_csv(
            '/home/MarkF/DivideConquer/Results/GPL570/All_Cancer/ICARUN/'
            'ica_robustness_metrics_independent_components_consensus.tsv', sep='\t', index_col=0)
        all_data.index = [f'{z}_big' for z in all_data.index]
        self.credibility_sets.append(f'big')
        self.credibility_small.append(all_data)
        self.credibility_big = pd.concat(self.credibility_small)
