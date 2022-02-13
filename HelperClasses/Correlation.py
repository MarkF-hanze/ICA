import numpy as np
import pandas as pd


class Correlation(object):
    """
    Class that calculates correlations between all columns between two dataframe
    """
    def __init__(self, df1, df2):
        """
        Input variable:
            df1: Dataframe containing ESes
            df2: Dataframe containing ESes
        """
        # Calculate the correlation
        self.df = df1.join(df2)
        self.correlation = np.corrcoef(self.df.values, rowvar=False)
        # Take the absolute
        self.correlation = np.absolute(self.correlation)
        # Set the names
        self.correlation = pd.DataFrame(self.correlation, columns=self.df.columns, index=self.df.columns)
        self.df1 = df1
        self.df2 = df2

    def get_merged_normall(self):
        """
        Get the merged dataframes
        """
        return self.df

    def get_half_correlation(self):
        """
        Get the correlation only between df1 and df2
        """
        return self.correlation.loc[self.df2.columns, self.df1.columns]

    def get_correlation(self):
        """
        Get the complete correlation
        """
        return self.correlation.copy()


