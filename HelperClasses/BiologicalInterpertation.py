import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools


class Histogram(object):
    def __init__(self, custom_colors):
        self.custom_colors = custom_colors
        self.used_colors = 0

    @staticmethod
    def clean_correlation(correlation, col_names):
        # Get the correlations and melt it
        subset_cor = correlation.loc[col_names, col_names]
        subset_cor = subset_cor.reset_index().melt(id_vars='index')
        # Remove sames values like ab and ba
        subset_cor = subset_cor.loc[
            pd.DataFrame(np.sort(subset_cor[['index', 'variable']], 1),
                         index=subset_cor.index).drop_duplicates(keep='first').index]
        # Remove the correlation to itself
        subset_cor = subset_cor[subset_cor['index'] != subset_cor['variable']]
        # X_axis name
        subset_cor['name'] = subset_cor['index'].str.split('_', expand=True)[1] + ' vs ' + \
                             subset_cor['variable'].str.split('_', expand=True)[1]
        subset_cor = subset_cor.sort_values(['name'], axis=0)
        # Remove the big vs big correlation because it is not needed in the plot
        subset_cor = subset_cor[subset_cor['name'] != 'big vs big']
        subset_cor['name2'] = np.arange(0, subset_cor.shape[0])
        return subset_cor

    @staticmethod
    def init_figures(amount):
        # Get the amount of rows and cols for a square plot
        rows = round(np.sqrt(amount))
        cols = round(np.sqrt(amount))
        # If there is a rounding error add another row
        if rows * cols < amount:
            rows += 1
        # Make the figure and if it is only 1 plot make it a numpy array (multiple plots will already be a np array)
        fig, axs = plt.subplots(rows, cols, sharey=True)
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        return fig, axs

    def set_colors(self, subset_cor):
        # Make the color bars
        colors = [[], []]
        # The used colormap
        cm = plt.get_cmap('tab20')
        # Loop over every bar
        for name in subset_cor['name']:
            # Get the 2 groups for this bar
            splitter = name.split(' vs ')
            for q, split in enumerate(splitter):
                # Check if the group has already been assigned a color and if not assign it (max 20 colors)
                if split not in self.custom_colors:
                    self.custom_colors[split] = cm(self.used_colors)
                    self.used_colors += 1
                # Append the color for each group
                colors[q].append(self.custom_colors[split])
        return colors

    def update_layout(self, fig, axs):
        # Remove x tick labels
        for ax in range(len(axs.ravel())):
            axs.ravel()[ax].set_xticks([])
        # Make a custom legend with the used colors (In total so can contain colors not in this specific plot)
        handles = []
        for name in self.custom_colors:
            handles.append(mpatches.Patch(color=self.custom_colors[name], label=name))
        # Place the legend to the right of the plot (Values may need to be changed depending on situation)
        lgd = fig.legend(handles=handles, shadow=True, fancybox=True, bbox_to_anchor=(1.4, 0.9))
        # Global x and y labels for the figure
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.ylabel("Pearsons correlation")
        return fig, axs

    def plot(self, plot_columns, correlation):
        fig, axs = self.init_figures(len(plot_columns))
        # Start looping the to be plotted columns
        for z, col_names in enumerate(plot_columns):
            # Get the smaller correlation of relevant columns
            subset_cor = self.clean_correlation(correlation, col_names)
            # Get the to be used colors
            colors = self.set_colors(subset_cor)
            # Make the color bars
            axs.ravel()[z].bar(subset_cor['name2'], (subset_cor['value'] / 10) * 9, color=colors[0])
            axs.ravel()[z].bar(subset_cor['name2'], (subset_cor['value'] / 10) * 1,
                               bottom=(subset_cor['value'] / 10) * 9, color=colors[1])
        fig, axs = self.update_layout(fig, axs)
        plt.show()
        # plt.savefig(f'{save_directory}/{name_file}.svg', dpi=1200,
        #            bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.clf()

    def get_colormap(self):
        return self.custom_colors


class MergeTwo(object):
    def __init__(self, correlation):
        self.original_cor = correlation.copy()
        self.correlation = correlation.copy()
        self._pivot_correlation()
        self.best_scores = {
            'Small Component': [],
            'Big Component': [],
            'Small2 Component': [],
            'Big2 Component': [],
            'Score': [],
        }
        self._get_scores()

    def _pivot_correlation(self):
        # Melt to reduce it to two columns
        self.correlation = self.correlation.reset_index().melt(id_vars='index').dropna()
        # Remove the big component from 1 column
        self.correlation = self.correlation[~self.correlation['index'].str.contains('big')]
        # Make columns with only the groups
        self.correlation['group 1'] = self.correlation['index'].str.split('_', expand=True)[1].values
        self.correlation['group 2'] = self.correlation['variable'].str.split('_', expand=True)[1].values
        # Remove lines going to the same group
        self.correlation = self.correlation[self.correlation['group 2'] != self.correlation['group 1']]
        # Get the max correlation from 1 ES to any and all of the groups
        # Example: EC1_group1 group 2 = max cor 0.35
        #          EC1_group1 group 3 = max cor 0.263
        #          EC1_group1 group big = max cor 0.71
        self.correlation = self.correlation[
            self.correlation.groupby(['index', 'group 2'])['value'].transform(max) == self.correlation['value']]

    def _get_scores(self):
        # TODO init test of er net zoveel len als groups zijn
        # Loop over every component except the one in big
        for component, comp_df in self.correlation.groupby('index'):
            # Get the highest big component correlation
            self.best_scores['Big Component'].append(
                comp_df[comp_df['variable'].str.contains('big')]['variable'].values[0])
            # Get the current small component
            self.best_scores['Small Component'].append(component)
            # Get the highest small component correlation
            self.best_scores['Small2 Component'].append(comp_df[~comp_df['variable'].str.contains('big')].sort_values(
                by='value', ascending=False)['variable'].values[0])
            # Get the comp_df for the second component
            comp_df_small2 = self.correlation[self.correlation['index'] == self.best_scores['Small2 Component'][-1]]
            # Get the big component highest with the second small component
            self.best_scores['Big2 Component'].append(
                comp_df_small2[comp_df_small2['variable'].str.contains('big')]['variable'].values[0])
            # Calculate the final score for this component, this is calculated as:
            # Max correlation component with the small set - max correlation with the big set
            self.best_scores['Score'].append(
                self.original_cor.loc[self.best_scores['Small Component'][-1], self.best_scores['Small2 Component'][-1]]
                - self.original_cor.loc[self.best_scores['Small Component'][-1], self.best_scores['Big Component'][-1]])

    def plot(self, plotter, plot_cutoff):
        # Turn best scores to a dataframe
        plot_df = pd.DataFrame.from_dict(self.best_scores, orient='index').transpose().sort_values(by='Score',
                                                                                                   ascending=False)
        # Remove the same values
        plot_df = plot_df.loc[
            pd.DataFrame(np.sort(plot_df[['Small Component', 'Small2 Component']], 1),
                         index=plot_df.index).drop_duplicates(keep='first').index]
        # TODO save file
        # df.to_csv(f'{save_directory}/Biological_int.csv')
        # Only plot the histograms with score bigger than cutoff
        # Set it in a list of list for the histogram
        plot_columns = plot_df[plot_df['Score'] > plot_cutoff].drop('Score', axis=1).iloc[:].values
        plotter.plot(plot_columns, self.original_cor)


class BigSmall(object):
    def __init__(self, correlation):
        self.correlation = correlation.copy()
        self.big_correlation = correlation.copy()
        self._pivot_correlation()

    def _pivot_correlation(self):
        self.big_correlation[self.big_correlation < 0.6] = np.nan
        # Get the components that are both in the big group and have a correlation with at least 1 small component
        # Only leave components that correlate with at least 2 others (bigger than 2 because it also correlated with
        # itself)
        self.subset = self.big_correlation.count(axis=1)
        self.subset = self.subset[self.subset > 2]

    def _get_big_components(self):
        # Only get the big components
        big_components = [i for i in self.big_correlation.columns if "big" in i]
        # Get the components in big and have a correlation of 0.6 with 2 components
        loop = set(big_components).intersection(set(self.subset.index))
        return loop

    def _make_component_df(self, component):
        # Get the big source and the highly correlated sources and drop the big component
        component_df = pd.DataFrame(self.big_correlation.loc[component].dropna().drop(component))
        # Make a column against what group the correlation is
        component_df["Group"] = component_df.reset_index()["index"].str.split(
            "_", expand=True).iloc[:, 1].values
        # Remove correlation with its own group (big correlation of 0.6 or higher with big)
        component_df = component_df[component_df["Group"] != component.split("_")[-1]]
        return component_df

    @staticmethod
    def multiple_reconstructed(df):
        # Get all how often a single group appears
        counts = df.groupby("Group").count()
        # Do some groups appear mutliple times?
        counts = counts[counts.iloc[:, 0] >= 2]
        # Return true if it is multiple connected
        return len(counts) > 0

    def get_combinations(self, component_df):
        # Look if there are multiple correlations coming from 1 set
        if self.multiple_reconstructed(component_df):
            # If this is the case get all possible combinations of 2 sets
            all_combinations = []
            for name, group in component_df.groupby("Group"):
                all_combinations.append(list(group.index))
            all_combinations = itertools.product(*all_combinations)
            all_combinations = [list(z) for z in all_combinations]
        else:
            # If this is not the case the only combination is all together
            all_combinations = [list(component_df.index)]
        return all_combinations

    @staticmethod
    def get_small_cor(component_df, small_components):
        return component_df.loc[small_components, :].drop("Group", axis=1).mean()

    def correlation_pairs(self, pairs):
        correlation_values = []
        for pair in pairs:
            #print(pair)
            correlation_values.append(self.correlation.loc[pair[0], pair[1]])
        return correlation_values

    def create_scores(self):
        # Dictionaries to save the results
        data = {}
        end_results = {}
        components = {}
        loop = self._get_big_components()
        for component in loop:
            # Only leave the relevant correlations
            component_df = self._make_component_df(component)
            # Get the unique groups this component correlates with
            groups = component_df["Group"].unique()
            # If it correlates with 2 or more unique groups
            if len(groups) >= 2:
                combinations = self.get_combinations(component_df)
                # Loop over very combination
                for z, small_components in enumerate(combinations):
                    # Take the mean of the remaining component, this is the distance from big to the smaller components
                    data[f"{component}_{z}"] = [self.get_small_cor(component_df, small_components)]
                    # Add all components that were used
                    components[f"{component}_{z}"] = list(small_components)
                    components[f"{component}_{z}"].append(component)
                    # Mean distance of all combinations of small components
                    small_pairs = itertools.combinations(small_components, 2)
                    data[f"{component}_{z}"].append(np.mean(self.correlation_pairs(small_pairs)))
                    # Best components have high correlation between big and small and low between small small
                    score = data[f"{component}_{z}"][0] - data[f"{component}_{z}"][1]
                    end_results[f"{component}_{z}"] = score.values[0]
        # Sort the end results to see the best components
        high_to_low = [k for k, v in sorted(end_results.items(), key=lambda item: item[1])][::-1]
        return [components[x] for x in high_to_low]

    def plot(self, plotter):
        # Dictionaries to save the results
        input_figure = self.create_scores()
        #pd.DataFrame(input_figure).to_csv(f'{save_directory}/EC_splitted.csv')
        plotter.plot(input_figure, self.correlation)
