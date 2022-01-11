import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from bokeh.io import export_png
from bokeh.plotting import figure, ColumnDataSource
from tqdm import tqdm
import numpy as np
import dcor
import itertools
import matplotlib.pyplot as plt

class Heatmap(object):
    def __init__(self, correlation, cut_off, saver):
        self.saver = saver
        self.clusters = []
        # Make variables
        self.cut_off = cut_off
        self.correlation = correlation.copy()
        # Make heatmap with only 0 or 1
        self.one_zero_correlation = correlation.copy()
        self.transform_correlation()
        # Make a dendogram to make the heatmap more interpretable
        self.col_order = self.make_linkage()

    def transform_correlation(self):
        self.one_zero_correlation[self.one_zero_correlation < self.cut_off] = 0
        self.one_zero_correlation[self.one_zero_correlation >= self.cut_off] = 1

    def make_linkage(self):
        # Get the values with more then 1 value to make it faster clustering with only 1 value doesnt make sense
        linkage_df = self.one_zero_correlation.sum()
        linkage_df = linkage_df[linkage_df > 1]
        # Make the linkage
        z = linkage(self.one_zero_correlation[linkage_df.index].values, method='ward',
                    optimal_ordering=True)
        # Add the linkage column that were removed earlier
        cols = [self.one_zero_correlation.columns[x] for x in leaves_list(z)]
        for column in self.one_zero_correlation:
            if column not in cols:
                cols.append(column)
        return cols

    def plot(self):
        df = self.one_zero_correlation.loc[self.col_order, self.col_order]
        df = df.iloc[:100, :100]
        # Color pallet
        #palette = ['#0000FF', '#00FFFF']
        palette = ['white', 'black']
        # Make the heatmap as bokeh figure
        x_as = []
        y_as = []
        colors = []
        for y in df.index:
            for x in df.columns:
                y_as.append(y)
                x_as.append(x)
                colors.append(palette[int(self.one_zero_correlation.loc[y, x])])
        # Make the dataframe as a source for plotting
        df = pd.DataFrame()
        df["x"], df["y"], df["colors"] = x_as, y_as, colors
        # Make the figure with layout
        p = figure(title="Categorical Heatmap",
                   x_range=df["x"].unique(), y_range=df["y"].unique(),
                   width=500, height=500)
        source = ColumnDataSource(df)
        p.rect("x", "y", color="colors", width=1, height=1, line_width=0.4, line_color="white", source=source,
               fill_color="colors")
        p.axis.visible = False
        p.toolbar.logo = None
        p.toolbar_location = None
        # Export it
        export_png(p, filename=f'{self.saver.get_path()}/Heatmap.png')

    def make_clusters(self):
        """
        Insert correlation matrix with only 0 and 1 with a certain cutoff
        """
        df = self.one_zero_correlation.copy()
        # Get every component that can be clustered
        df['Count'] = df.sum(axis=1)
        df = df.sort_values('Count', axis=0, ascending=False)
        # Save the intermediate results
        temp_df = df.copy()
        # Loop over the values
        for index, row in df.iterrows():
            # Check if values are already clustered
            temp_df['Count'] = temp_df.sum(axis=0)
            # If it still has more than 1 value that it correlates with
            if temp_df.loc[index, 'Count'] > 1:
                # Get the components that have a one in this components row
                components = list(row[row > 0].index)[:-1]
                # Append the components to the value and put it on zero
                self.clusters.append(components)
                temp_df[components] = 0

    def set_cluster(self, cluster):
        self.clusters = cluster

    def make_negative(self, vector):
        return -vector

    def do_nothing(self, vector):
        return vector

    def get_original_distance(self,big_comp, small_components):
        distance = 0
        # Get all the original distances from small to big
        for component in small_components:
            original_distance = dcor.distance_correlation(component, big_comp)
            # If the correlation is higher make it the best components
            if original_distance > distance:
                distance = original_distance
        return distance

    def get_new_distances(self, big_comp, small_components):
        distance = 0
        # Make every combination of adding or subtracting
        operations = (self.do_nothing, self.make_negative)
        operations = itertools.product(operations, repeat=len(small_components))
        for operation in operations:
            # Add all small components together (some are made negative so this is subtracting)
            outcome = np.zeros((small_components[0].shape[0]))
            for component, op in zip(small_components, operation):
                outcome = outcome.ravel() + op(np.array(component)).ravel()
            # Calculate the distance of this components to the big component
            new_distances = dcor.distance_correlation(outcome, big_comp)
            # If it is bigger add it
            if new_distances > distance:
                distance = new_distances
        return distance

    def merge_clusters(self, df):
        xs = []
        ys = []
        # Get the amount of small sets in the estimated sources
        small_groups = [z[1] for z in df.columns.str.split('_')]
        small_groups = set(small_groups)
        small_groups = [f'_{z}' for z in small_groups if z != 'big']
        # Loop over every cluster
        for cluster in tqdm(self.clusters):
            # For now just skip it if multiple estimated components come from the same set
            if len(set([x.split('_')[1] for x in cluster])) != len(cluster):
                continue
            # If the cluster has more than 2 small clustered
            if len(cluster) > 2:
                # Get the small components for every small set, if the small set doesnt exist add zeros
                small_components = []
                for group in small_groups:
                    comp = [s for s in cluster if group in s]
                    if len(comp) > 0:
                        value = df[comp]
                    else:
                        value = np.zeros((df.shape[0]))
                    small_components.append(value)
                # Check for every small component the distance to the big component
                # First check if there is a big component
                if len([s for s in cluster if '_big' in s]) > 0:
                    # Get the big component
                    big_comp = df[[s for s in cluster if '_big' in s][0]]
                    max_original_distances = self.get_original_distance(big_comp, small_components)
                    # Check every linear combination of the small components
                    max_new_distances = self.get_new_distances(big_comp, small_components)

                    xs.append(max_original_distances)
                    ys.append(max_new_distances)
        self.plot_merged(xs, ys)

    def plot_merged(self, xs, ys):
        # Make the scatter plot
        plt.clf()
        plt.scatter(xs, ys, marker="+", color="blue")
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Original distance correlation')
        plt.ylabel('New distance correlation')
        plt.savefig(f'{self.saver.get_path()}/Fading_clustered.svg')
        plt.clf()



