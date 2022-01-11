from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from tqdm import tqdm

from HelperClasses.Correlation import Correlation
from HelperClasses.LoadData import LoadICARuns


class CompareTactics(object):
    def __init__(self, save_path):
        self.save_path = save_path
        self.directories = None
        # Only compatible with 2,3 and 4 checks
        self.line_styles = {
            '2': '-',
            '3': '--',
            '4': '-.'
        }
        self.colors = {'Clustered': '#0000FF',
                  'Random': '#00FFFF',
                  'big': '#FF7F00'}
        self.files = {}

    def load_data(self, directories):
        self.directories = directories
        for directory in self.directories:
            loader = LoadICARuns(
                directory[1],
                '/home/MarkF/DivideConquer/Results/2000_Samples_Experiment/Clustered_vs_Random_Experiment/'
                'ICARUN_ALL/ica_independent_components_consensus.tsv')
            self.files[directory[0]] = {'path': directory[1], 'loader': loader}
            self.files[directory[0]]['correlation'] = Correlation(loader.get_merged_small(), loader.get_sample_data())

    def check_distribution(self):
        fig = plt.figure(constrained_layout=True)
        # Get every point from 0 to 0.9
        xs = np.linspace(0.0, 0.9, 10)
        for split in self.directories:
            # Load the maximun correlation
            df = self.files[split[0]]['correlation'].get_half_correlation()
            ys = []
            for x in xs:
                ys.append(df[df >= x].dropna(how="all").shape[0])
            # Plot how many component occur for every cutoff correlation point
            plt.plot(xs, ys, color=self.colors[split[0].split('_')[1]],
                     linestyle=self.line_styles[split[0].split('_')[0]],
                     label=split[0].replace('_', ' '))
        plt.title(f'Amount of big components that correlate with small n = {df.shape[0]}')
        plt.xlabel("Pearson's cutoff")
        plt.ylabel('Consensus count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/Pearson_distribution_Cutoff.svg', dpi=1200)
        #plt.show()

    def get_credibility(self):
        # Get the big value from a random set
        test = {}
        small_df, df_big, sets = self.files['2_Clustered']['loader'].get_credibility()
        # Get the big index credibility values
        index = [z for z in df_big.index if '_big' in z]
        credibility_values_big = df_big.loc[index, 'credibility index'].values
        # Put it in a dataframe for plotting
        plot_df = {'big_big': credibility_values_big}
        # Put it in a list for testing later
        credibility_values_big = [sum(x >= .8 for x in credibility_values_big),
                                  sum(x < .8 for x in credibility_values_big)]
        save_df = []
        save_df_1 = []
        # Load every the credibility index
        for split in self.directories:
            small_df, df_big, sets = self.files[split[0]]['loader'].get_credibility()
            # Make the consensus variables
            consensus_index = self.get_consensus(split[0])
            # Save it
            save_df_1.append([split[0], len(set(consensus_index))])
            # Get the values not in big and in the consensus
            credibility_values = df_big.loc[consensus_index, 'credibility index'].values
            # Save it for later use
            plot_df[split[0]] = credibility_values
            # Fisher exact cutoff and put it in the table put this table against a table
            test[split[0]] = [sum(x >= .8 for x in credibility_values),
                              sum(x < .8 for x in credibility_values)]
            table = [credibility_values_big, test[split[0]]]
            table = np.array(table)
            save_df.append([split[0], 'Big', stats.fisher_exact(table, alternative='less')[1]])
        # What different distributions to check against each other
        checking = [['2_Clustered', '3_Clustered'], ['2_Clustered', '4_Clustered'], ['3_Clustered', '4_Clustered'],
                    ['2_Random', '3_Random'], ['2_Random', '4_Random'], ['3_Random', '4_Random'],
                    ['2_Random', '2_Clustered'], ['3_Random', '3_Clustered'], ['4_Clustered', '4_Random']]
        # Add these test results also with the fisher excact test
        for check in checking:
            table = []
            for tab in check:
                table.append(test[tab])
            save_df.append([check[0], check[1], stats.fisher_exact(table, alternative='less')[1]])
        # Dictionaries to dataframe and safe them
        save_df = pd.DataFrame(save_df, columns=['Group 1', 'Group 2', 'Fisher exact p_value'])
        save_df_1 = pd.DataFrame(save_df_1, columns=['Split', 'Consensus count'])
        # Plot the consensus counts
        estimated_count = save_df_1.copy()
        estimated_count[['Number \n of splits', 'Split type']] = estimated_count['Split'].str.split('_', expand=True)
        sns.lineplot(data=estimated_count, x='Number \n of splits', y='Consensus count', hue='Split type', marker='o',
                     palette=self.colors)
        plt.tight_layout()
        plt.savefig('Results/Random_VS_Clustered/Estimated_sources_count.svg', dpi=1200)
        #save_df.to_csv('Results//Random_VS_Clustered/Credibility_distribution_correaltions.csv', index=False)
        print('--------------------------------------------')
        print(save_df)
        print(save_df_1)
        #save_df_1.to_csv('Results/Random_VS_Clustered/Estimated_sources_count.csv', index=False)
        # Make a dataframe for plotting
        plot_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in plot_df.items()]))
        plot_df = pd.melt(plot_df, value_name='Credibility Index')
        plot_df[['number', 'tactic']] = plot_df['variable'].str.split('_', expand=True)
        # Make the violin plot without the big one for now becuase it looks better
        plot_df = plot_df[plot_df['number'] != 'big']
        self.plot_violin_cred(plot_df)

    def plot_violin_cred(self, df):
        plt.clf()
        sns.violinplot(data=df, y='Credibility Index', x='number', hue='tactic', palette=self.colors,
                       cut=0, split=True)
        plt.xlabel('Number of splits')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title('Distribution of the consensus estimated components')
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/Credibility_distribution_correlations.svg', dpi=1200)
        #plt.show()
        plt.clf()

    def get_consensus(self, split):
        _, df, _ = self.files[split]['loader'].get_credibility()
        correlation = self.files[split]['correlation'].get_correlation()
        # All small components
        index = [z for z in correlation.columns if 'big' not in z]
        df_small = df.loc[index, :]
        # Get the distance correlation between all combinations of small components
        correlation = correlation.loc[df_small.index, df_small.index]
        # Melt it and remove correlation between the same group
        correlation = correlation.reset_index().melt(id_vars='index').dropna()
        correlation = correlation[correlation['index'].str.split(
            '_', expand=True)[1] != correlation['variable'].str.split('_', expand=True)[1]]
        # Only leave the components that correlated with at least 1 other component (bigger than 1 cause diagonal)
        drop_columns = []
        correlation_cutoff = 0.6
        estimated_sources = list(correlation[correlation['value'] > correlation_cutoff]['index'])
        # Loop until no estimated sources is left with a correlation larger than 0
        while len(estimated_sources) > 0:
            # Get the components that correlate with the first estimated source in the list
            estimated_source_df = correlation[(correlation['value'] > correlation_cutoff) &
                                              (correlation['index'] == estimated_sources[0])]

            all_correlated = list(set(estimated_source_df[['index', 'variable']].values.ravel()))
            # Remove these components from the estimated sources
            estimated_sources = [z for z in estimated_sources if z not in all_correlated]
            # Only leave the column with the highest credibility index
            drop_columns.extend(list(df_small.loc[all_correlated, :].sort_values(
                by='credibility index', ascending=False).iloc[1:].index))
        test = correlation.copy()
        test = test[~test['index'].isin(drop_columns)]
        test = test[~test['variable'].isin(drop_columns)]
        index = [x for x in index if x not in drop_columns]
        return index

    def consensus_vs_correlation(self):
        sns.set(font_scale=1.2, style="whitegrid")
        # Get the sample dataset credibility index (Is the last added dataset)
        end_df, _, _ = self.files['2_Clustered']['loader'].get_credibility()
        end_df = pd.DataFrame(end_df[-1]['credibility index'])
        save_df = []
        for split in self.directories:
            # Get the correlation and get the big and small columns
            correlation = self.files[split[0]]['correlation'].get_half_correlation()
            # Take the max for every big
            correlation = correlation.max(axis=1)
            correlation.name = f'Correlation_{split[0]}'
            correlation = pd.DataFrame(correlation)
            # Make it in one dataframe for testing
            end_df = end_df.join(correlation, how='inner')
            # Do the spearman correlaation and save the results
            cor, p = stats.spearmanr(end_df['credibility index'], end_df[f'Correlation_{split[0]}'])
            save_df.append([split[0], cor, p])
        # Transform the dataframe for plotting
        end_df = end_df.reset_index()
        end_df = end_df.melt(id_vars=['index', 'credibility index'])
        end_df['group'] = [x.split('_', 1)[-1] for x in end_df['variable']]
        end_df[['split', 'group']] = end_df['group'].str.split('_', expand=True)
        end_df = end_df.rename(columns={"value": "Spearman correlation"})
        # Start making the plot
        g = sns.FacetGrid(end_df, col="split", row='group', margin_titles=True)
        g.map(sns.scatterplot, 'credibility index', "Spearman correlation", alpha=.7)
        plt.savefig(f'{self.save_path}/Correlation_vs_Credibility.svg', dpi=1200)
        #plt.show()
        plt.clf()
        save_df = pd.DataFrame(save_df, columns=['Group', 'Spearman correlation', 'Spearman p value'])
        print(save_df)
        #save_df.to_csv('Results/Correlation_vs_Credibility.csv', index=False)
        sns.set(font_scale=1)

    def consensus_big(self):
        # Test the maximun correlation of the sample set with different splitting methods
        sns.set(font_scale=2.4, style='whitegrid')
        fig = plt.figure(figsize=(15, 5))
        test = {}
        save_df = []

        checking = [['2_Clustered', '3_Clustered'], ['2_Clustered', '4_Clustered'], ['3_Clustered', '4_Clustered'],
                    ['2_Random', '3_Random'], ['2_Random', '4_Random'], ['3_Random', '4_Random'],
                    ['2_Clustered', '2_Random'], ['3_Clustered', '3_Random'], ['4_Clustered', '4_Random']]
        for split in self.directories:
            correlation = self.files[split[0]]['correlation'].get_half_correlation()
            correlation = correlation.max(axis=1)
            test[split[0]] = correlation.values
            print(split[0])
        # Test the distibution with a Welsh-t-test
        for check in tqdm(checking):
            save_df.append([check[0], check[1], stats.ttest_ind(test[check[0]], test[check[1]],
                                                                equal_var=True, alternative='greater',
                                                                permutations=100_000)[1]])
        # Save the test results
        save_df = pd.DataFrame(save_df, columns=['Group 1', 'Group 2', 'Welsh p_value'])
        #save_df.to_csv('Results/Pearson_distribution_test.csv', index=False)
        # Start plotting the distributions in a violine plot
        plot_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in test.items()]))
        plot_df = pd.melt(plot_df, value_name='Pearson correlation')
        plot_df[['number', 'tactic']] = plot_df['variable'].str.split('_', expand=True)
        sns.violinplot(data=plot_df, y='Pearson correlation', x='number', hue='tactic', palette=self.colors,
                       cut=2, split=True)
        plt.title(f"Density plot of highest correlation for every estimated source \n in the sample data")
        plt.xlabel("Number of splits")
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        #plt.show()
        plt.savefig(f'{self.save_path}/Pearson_distribution_KDE.svg', dpi=1200, bbox_inches='tight')
        #
        sns.set(font_scale=1)

    def check_citrus(self):
        # output_file(filename=f"{save_directory}/citrusPlot.html")
        # Remove diagonal and values smaller than cutoff
        test = {}
        for split in self.directories:
            correlation = self.files[split[0]]['correlation'].get_correlation()
            #TODO also put this in the other melts
            for i in np.arange(correlation.shape[0]):
                correlation.iloc[i, i] = np.nan
            # Melt the dataframe to 3 columns
            lines = correlation.reset_index().melt(id_vars='index').dropna()
            # Drop the duplicates that like ab ba because correlation was mirrored
            lines = lines[lines['index'].str.split('_', expand=True)[1] != lines['variable'].str.split('_', expand=True)[1]]
            lines = lines.loc[
                pd.DataFrame(np.sort(lines[['index', 'variable']], 1), index=lines.index).drop_duplicates(
                    keep='first').index]
            # Drop the big components for the test
            test[split[0]] = lines[(~lines['index'].str.contains('big')) & (~lines['variable'].str.contains('big'))]
            test[split[0]] = [sum(x >= .6 for x in test[split[0]]['value']), sum(x < .6 for x in test[split[0]]['value'])]
            # See how many lines go to big
            big = lines[(lines['index'].str.contains('big')) | (lines['variable'].str.contains('big'))]
            big['index'] = big['index'].str.split('_', expand=True)[1]
            big['variable'] = big['variable'].str.split('_', expand=True)[1]
            big = big[big['value'] >= .6]
            big = big.groupby(['index', 'variable']).count()
            #big.to_csv(f'Results/CitrusCount_{split}.csv', index=True)
            print(big)
        checking = [['2_Clustered', '2_Random'], ['3_Clustered', '3_Random'], ['4_Clustered', '4_Random']]
        save_df = []
        for check in checking:
            table = np.array([test[check[0]], test[check[1]]])
            save_df.append([check[0], check[1], stats.fisher_exact(table, alternative='less')[1]])
        save_df = pd.DataFrame(save_df, columns=['Group 1', 'Group 2', 'Fisher exact p_value'])
        print(save_df)
        #save_df.to_csv('Results/CitrusCheck.csv', index=False)


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    compare = CompareTactics('Results/Random_VS_Clustered')
    directories = [('2_Clustered', '/home/MarkF/DivideConquer/Results/2000_Samples_Experiment/Clustered_vs_Random_Experiment/Clustered_Splits/2_Split'),
                   ('3_Clustered', '/home/MarkF/DivideConquer/Results/2000_Samples_Experiment/Clustered_vs_Random_Experiment/Clustered_Splits/3_Split'),
                   ('4_Clustered', '/home/MarkF/DivideConquer/Results/2000_Samples_Experiment/Clustered_vs_Random_Experiment/Clustered_Splits/4_Split'),
                   ('2_Random', '/home/MarkF/DivideConquer/Results/2000_Samples_Experiment/Clustered_vs_Random_Experiment/Random_Splits/2_Split'),
                   ('3_Random', '/home/MarkF/DivideConquer/Results/2000_Samples_Experiment/Clustered_vs_Random_Experiment/Random_Splits/3_Split'),
                   ('4_Random', '/home/MarkF/DivideConquer/Results/2000_Samples_Experiment/Clustered_vs_Random_Experiment/Random_Splits/4_Split')
                   ]
    compare.load_data(directories)
    compare.check_citrus()
    compare.get_credibility()
    compare.consensus_big()
    compare.check_distribution()
    compare.consensus_vs_correlation()
