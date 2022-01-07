import numpy as np
import pandas as pd


class Correlation(object):
    def __init__(self, df1, df2):
        self.df = df1.join(df2)
        self.correlation = np.corrcoef(self.df.values, rowvar=False)
        self.correlation = np.absolute(self.correlation)
        self.correlation = pd.DataFrame(self.correlation, columns=self.df.columns, index=self.df.columns)
        self.df1 = df1
        self.df2 = df2

    def get_merged_normall(self):
        return self.df

    def get_half_correlation(self):
        return self.correlation.loc[self.df2.columns, self.df1.columns]

    def get_correlation(self):
        return self.correlation.copy()


