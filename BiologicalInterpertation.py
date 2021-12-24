import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class MergeTwo(object):
    def __init__(self, correlation):
        # Melt to reduce it to two columns
        df = correlation.reset_index().melt(id_vars='index').dropna()
        # remove correlation between the same group of splits
        # df = df[df['index'].str.split('_', expand=True)[1] != df['variable'].str.split('_', expand=True)[1]]
        df = df[~df['index'].str.contains('big')]
        df['group 1'] = df['index'].str.split('_', expand=True)[1].values
        df['group 2'] = df['variable'].str.split('_', expand=True)[1].values
        df = df[df['group 2'] != df['group 1']]
        df = df[df.groupby(['index', 'group 2'])['value'].transform(max) == df['value']]

        best_scores = {
            'Small Component': [],
            'Big Component': [],
            'Small2 Component': [],
            'Big2 Component': [],
            'Score': [],
        }
        # TODO init test of er net zoveel len als groups zijn
        for component, comp_df in df.groupby('index'):
            best_scores['Big Component'].append(comp_df[comp_df['variable'].str.contains('big')]['variable'].values[0])
            best_scores['Small Component'].append(component)
            best_scores['Small2 Component'].append(comp_df[~comp_df['variable'].str.contains('big')].sort_values(
                by='value', ascending=False)['variable'].values[0])
            df2 = df[df['index'] == best_scores['Small2 Component'][-1]]
            best_scores['Big2 Component'].append(df2[df2['variable'].str.contains('big')]['variable'].values[0])

            best_scores['Score'].append(correlation.loc[best_scores['Small Component'][-1],
                                               best_scores['Small2 Component'][-1]] - correlation.loc[
                                   best_scores['Small Component'][-1],
                                   best_scores['Big Component'][-1]])

        df = pd.DataFrame.from_dict(best_scores, orient='index').transpose().sort_values(by='Score', ascending=False)
        # Remove the same values
        df = df.loc[
            pd.DataFrame(np.sort(df[['Small Component', 'Small2 Component']], 1), index=df.index).drop_duplicates(
                keep='first').index]
       # df.to_csv(f'{save_directory}/Biological_int.csv')
        # Bigger than 0.5 count it
        df = df[df['Score'] > 0.5]
        color_mapper = self.plot_histogram(correlation, df.drop('Score', axis=1).iloc[:].values, "Biological_int")
        return color_mapper

    def plot_histogram(self, correlation, columns, name_file):
        rows = round(np.sqrt(len(columns)))
        cols = round(np.sqrt(len(columns)))
        if rows * cols < len(columns):
            rows += 1
        fig, axs = plt.subplots(rows, cols, sharey=True)
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        color_mapper = {'big': 'black'}
        color_i = 0
        for z, col_name in enumerate(columns):
            # Get the correlations
            df = correlation.loc[col_name, col_name]
            df = df.reset_index().melt(id_vars='index')
            # Remove sames values
            df = df.loc[
                pd.DataFrame(np.sort(df[['index', 'variable']], 1), index=df.index).drop_duplicates(keep='first').index]
            # Remove the duplicates
            df = df[df['index'] != df['variable']]
            # Compare
            df['name'] = df['index'].str.split('_', expand=True)[1] + ' vs ' + \
                         df['variable'].str.split('_', expand=True)[1]
            df = df.sort_values(['name'], axis=0)
            df = df[df['name'] != 'big vs big']
            # Make the color bars
            colors = [[], []]
            cm = plt.get_cmap('tab20')
            for name in df['name']:
                splitter = name.split(' vs ')
                for q, split in enumerate(splitter):
                    if split not in color_mapper:
                        color_mapper[split] = cm(color_i)
                        color_i += 1
                    colors[q].append(color_mapper[split])
            df['name2'] = np.arange(0, df.shape[0])
            axs.ravel()[z].bar(df['name2'], (df['value'] / 10) * 9, color=colors[0])
            axs.ravel()[z].bar(df['name2'], (df['value'] / 10) * 1, bottom=(df['value'] / 10) * 9, color=colors[1])
        # Remove ticklabels
        for ax in range(len(axs.ravel())):
            axs.ravel()[ax].set_xticks([])
        # Legend
        handles = []
        for name in color_mapper:
            handles.append(mpatches.Patch(color=color_mapper[name], label=name))
        # axs[0][-1].legend(handles=handles, shadow=True, fancybox=True, bbox_to_anchor=(2.1, 1))
        lgd = fig.legend(handles=handles, shadow=True, fancybox=True, bbox_to_anchor=(1.4, 0.9))
        # Layout
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.ylabel("Pearsons correlation")
        # plt.tight_layout()
        plt.show()
        #plt.savefig(f'{save_directory}/{name_file}.svg', dpi=1200,
        #            bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.clf()
        sys.exit()
        return color_mapper