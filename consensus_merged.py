from consensus import load_data, calculate_correlation, correlation_with_cutoff, load_credibility_cancer, \
    load_credibility
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import sys
from tqdm import tqdm

line_styles = {
    '2': '-',
    '3': '--',
    '4': '-.'
}
# colors = {'Clustered': 'mediumblue',
#           'Random': 'lightblue'}

colors = {'Clustered': '#0000FF',
          'Random': '#00FFFF',
          'big': '#FF7F00'}


def check_distribution(dictionairy):
    fig = plt.figure(constrained_layout=True)
    xs = np.linspace(0.0, 0.9, 10)
    for split in dictionairy:
        df = dictionairy[split]['Half_Correlation']
        ys = []
        for x in xs:
            ys.append(df[df >= x].dropna(how="all").shape[0])
        plt.plot(xs, ys, color=colors[split.split('_')[1]], linestyle=line_styles[split.split('_')[0]],
                 label=split.replace('_', ' '))
    plt.title(f'Amount of big components that correlate with small n = {df.shape[0]}')
    plt.xlabel("Pearson's cutoff")
    plt.ylabel('Consensus count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Results/Pearson_distribution_Cutoff.svg', dpi=1200)


def get_credibility(directory):
    bw_adjust = 0.5
    test = {}
    small_df, df_big, sets = load_credibility(directory['2_Clustered']['path'])
    index = [z for z in df_big.index if '_big' in z]
    credibility_values_big = df_big.loc[index, 'credibility index'].values
    plot_df = {'big_big': credibility_values_big}
    credibility_values_big = [sum(x >= .8 for x in credibility_values_big),
                              sum(x < .8 for x in credibility_values_big)]
    save_df = []
    save_df_1 = []
    for split in directory:
        small_df, df_big, sets = load_credibility(directory[split]['path'])
        directory[split]['Credibility'] = df_big
        consensus_index = get_consensus(directory[split])
        save_df_1.append([split, len(set(consensus_index))])
        # Get the values not in big and in the consensus
        credibility_values = df_big.loc[consensus_index, 'credibility index'].values
        plot_df[split] = credibility_values
        test[split] = [sum(x >= .8 for x in credibility_values), sum(x < .8 for x in credibility_values)]
        table = [credibility_values_big, test[split]]
        table = np.array(table)
        save_df.append([split, 'Big', stats.fisher_exact(table, alternative='less')[1]])
    #     sns.kdeplot(x=credibility_values, color=colors[split.split('_')[1]],
    #                 linestyle=line_styles[split.split('_')[0]],
    #                 label=f"{split.replace('_', ' ')} \n n={len(credibility_values)}",
    #                 fill=False, clip=(0, 1), bw_adjust=bw_adjust)
    checking = [['2_Clustered', '3_Clustered'], ['2_Clustered', '4_Clustered'], ['3_Clustered', '4_Clustered'],
                ['2_Random', '3_Random'], ['2_Random', '4_Random'], ['3_Random', '4_Random'],
                ['2_Random', '2_Clustered'], ['3_Random', '3_Clustered'], ['4_Clustered', '4_Random']]
    for check in checking:
        table = []
        for tab in check:
            table.append(test[tab])
        save_df.append([check[0], check[1], stats.fisher_exact(table, alternative='less')[1]])
    save_df = pd.DataFrame(save_df, columns=['Group 1', 'Group 2', 'Fisher exact p_value'])
    save_df_1 = pd.DataFrame(save_df_1, columns=['Split', 'Consensus count'])
    save_df_1.to_csv('Results/Estimated_sources_count.csv', index=False)
    # column = [z for z in correlation.columns if '_big' in z]
    # credibility_values = df_big.loc[column, 'credibility index'].values
    # sns.kdeplot(x=credibility_values, color='#FF7F00',
    #             linestyle='dotted', label=f"All Credibility \n  n={len(credibility_values)}",
    #             fill=False, clip=(0, 1), bw_adjust=bw_adjust)
    # plt.xlim(0, 1)
    plot_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in plot_df.items()]))
    plot_df = pd.melt(plot_df, value_name='Credibility Index')
    plot_df[['number', 'tactic']] = plot_df['variable'].str.split('_', expand=True)
    plot_violin_cred(plot_df, False)
    plot_df = plot_df[plot_df['number'] != 'big']
    plot_violin_cred(plot_df, True)
    save_df.to_csv('Results/Credibility_distribution_correaltions.csv', index=False)
    consensus_vs_correlation(directory)

def plot_violin_cred(df, split):
    sns.violinplot(data=df, y='Credibility Index', x='number', hue='tactic', palette=colors,
                   cut=0, split=split)
    plt.xlabel('Number of splits')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title('Distribution of the consensus estimated components')
    plt.tight_layout()
    plt.savefig(f'Results/Credibility_distribution_correaltions_{split}.svg', dpi=1200)
    plt.clf()

def get_consensus(split_dictionairy):
    df = split_dictionairy['Credibility']
    correlation = split_dictionairy['Correlation']
    # All small components
    index = [z for z in correlation.columns if '_s' in z]
    df_small = df.loc[index, :]
    # All big components
    # Get the distance correlation between all combinations of small components
    correlation = correlation.loc[df_small.index, df_small.index]
    lines = correlation.reset_index().melt(id_vars='index').dropna()
    lines = lines[lines['index'].str.split('_', expand=True)[1] != lines['variable'].str.split('_', expand=True)[1]]
    # Only leave the components that correlated with at least 1 other component (bigger than 1 cause diagonal)
    credibility_df = pd.DataFrame()
    drop_columns = []
    correlation_cutoff = 0.6
    estimated_sources = list(lines[lines['value'] > correlation_cutoff]['index'])
    # TODO gaat dit goed?
    while len(estimated_sources) > 0:
        # Get the components that the estimated source correlates with
        estimated_source_df = lines[(lines['value'] > correlation_cutoff) & (lines['index'] == estimated_sources[0])]
        all_correlated = list(set(estimated_source_df[['index', 'variable']].values.ravel()))
        # Remove these components from the estimated sources
        estimated_sources = [z for z in estimated_sources if z not in all_correlated]
        # Only leave the column with the highest credibility index
        drop_columns.extend(list(df_small.loc[all_correlated, :].sort_values(
            by='credibility index', ascending=False).iloc[1:].index))
    index = [x for x in index if x not in drop_columns]
    return index


def consensus_vs_correlation(directory):
    sns.set(font_scale=1.2)
    end_df = pd.DataFrame(directory['2_Clustered']['Credibility']['credibility index'])
    save_df = []
    for split in directory:
        correlation = directory[split]['Correlation']
        column = [z for z in correlation.columns if '_big' in z]
        index = [z for z in correlation.columns if z not in column]
        correlation = correlation.loc[index, column].max(axis=0)
        correlation.name = f'Correlation_{split}'
        correlation = pd.DataFrame(correlation)
        test_df = pd.DataFrame(directory[split]['Credibility']['credibility index'])
        test_df = test_df.join(correlation, how='inner')
        cor, p = stats.spearmanr(test_df['credibility index'], test_df[f'Correlation_{split}'])
        save_df.append([split, cor, p])
        end_df = end_df.join(correlation, how='inner')
    end_df = end_df.reset_index()
    end_df = end_df.melt(id_vars=['index', 'credibility index'])
    end_df['group'] = [x.split('_', 1)[-1] for x in end_df['variable']]
    end_df[['split', 'group']] = end_df['group'].str.split('_', expand=True)
    end_df = end_df.rename(columns={"value": "Pearson's correlation"})
    g = sns.FacetGrid(end_df, col="split", row='group', margin_titles=True)
    g.map(sns.scatterplot, 'credibility index', "Pearson's correlation", alpha=.7)
    plt.savefig(f'Results/Correlation_vs_Credibility.svg', dpi=1200)
    plt.clf()
    save_df = pd.DataFrame(save_df, columns=['Group', 'Spearman correlation', 'Spearman p value'])
    save_df.to_csv('Results/Correlation_vs_Credibility.csv', index=False)
    sns.set(font_scale=1)


def consensus_big(dictionairy):
    sns.set(font_scale=2.4)
    fig = plt.figure(figsize=(15, 5))
    test = {}
    save_df = []
    checking = [['2_Clustered', '3_Clustered'], ['2_Clustered', '4_Clustered'], ['3_Clustered', '4_Clustered'],
                ['2_Random', '3_Random'], ['2_Random', '4_Random'], ['3_Random', '4_Random'],
                ['2_Clustered', '2_Random'], ['3_Clustered', '3_Random'], ['4_Clustered', '4_Random']]
    for split in dictionairy:
        correlation = dictionairy[split]['Half_Correlation']
        correlation = correlation.max(axis=1)
        test[split] = correlation.values
        # sns.kdeplot(x=correlation.values, color=colors[split.split('_')[1]], linestyle=line_styles[split.split('_')[0]],
        #             label=split.replace('_', ' '))
    for check in tqdm(checking):
        save_df.append([check[0], check[1], stats.ttest_ind(test[check[0]], test[check[1]],
                                                            equal_var=True, alternative='greater',
                                                            permutations=100_000)[1]])
    plot_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in test.items()]))
    plot_df = pd.melt(plot_df, value_name='Pearson correlation')
    plot_df[['number', 'tactic']] = plot_df['variable'].str.split('_', expand=True)
    sns.violinplot(data=plot_df, y='Pearson correlation', x='number', hue='tactic', palette=colors,
                   cut=2, split=True)
    save_df = pd.DataFrame(save_df, columns=['Group 1', 'Group 2', 'Welsh p_value'])
    plt.title(f"Density plot of highest correlation for every estimated source \n in the sample data")
    plt.xlabel("Number of splits")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(f'Results/Pearson_distribution_KDE.svg', dpi=1200, bbox_inches='tight')
    save_df.to_csv('Results/Pearson_distribution_test.csv', index=False)
    sns.set(font_scale=1)


def check_citrus(dictionairy):
    # output_file(filename=f"{save_directory}/citrusPlot.html")
    # Remove diagonal and values smaller than cutoff
    test = {}
    for split in dictionairy:
        correlation = dictionairy[split]['Correlation']
        correlation.values[[np.arange(correlation.shape[0])] * 2] = np.nan
        # Melt the dataframe to 3 columns
        lines = correlation.reset_index().melt(id_vars='index').dropna()
        # Drop the duplicates that like ab ba because correlation was mirrored
        lines = lines[lines['index'].str.split('_', expand=True)[1] != lines['variable'].str.split('_', expand=True)[1]]
        lines = lines.loc[
            pd.DataFrame(np.sort(lines[['index', 'variable']], 1), index=lines.index).drop_duplicates(
                keep='first').index]
        # Drop the big components for the test
        test[split] = lines[(~lines['index'].str.contains('big')) & (~lines['variable'].str.contains('big'))]
        test[split] = [sum(x >= .6 for x in test[split]['value']), sum(x < .6 for x in test[split]['value'])]
        # See how many lines go to big
        big = lines[(lines['index'].str.contains('big')) | (lines['variable'].str.contains('big'))]
        big['index'] = big['index'].str.split('_', expand=True)[1]
        big['variable'] = big['variable'].str.split('_', expand=True)[1]
        big = big[big['value'] >= .6]
        big = big.groupby(['index', 'variable']).count()
        big.to_csv(f'Results/CitrusCount_{split}.csv', index=True)
    checking = [['2_Clustered', '2_Random'], ['3_Clustered', '3_Random'], ['4_Clustered', '4_Random']]
    save_df = []
    for check in checking:
        table = np.array([test[check[0]], test[check[1]]])
        save_df.append([check[0], check[1], stats.fisher_exact(table, alternative='less')[1]])
    save_df = pd.DataFrame(save_df, columns=['Group 1', 'Group 2', 'Fisher exact p_value'])
    save_df.to_csv('Results/CitrusCheck.csv', index=False)


if __name__ == "__main__":
    files = {}
    directories = [('2_Clustered', '/home/MarkF/DivideConquer/Results/Math_Clustered/2_Split/'),
                   ('3_Clustered', '/home/MarkF/DivideConquer/Results/Math_Clustered/3_Split/'),
                   ('4_Clustered', '/home/MarkF/DivideConquer/Results/Math_Clustered/4_Split/'),
                   ('2_Random', '/home/MarkF/DivideConquer/Results/MathExperiment/2_Split/One_Normalized'),
                   ('3_Random', '/home/MarkF/DivideConquer/Results/MathExperiment/3_Split/'),
                   ('4_Random', '/home/MarkF/DivideConquer/Results/MathExperiment/4_Split/')
                   ]
    for directory in directories:
        files[directory[0]] = {'path': directory[1]}
        files[directory[0]]['Small data'], files[directory[0]]['Big data'], files[directory[0]][
            'Lookup columns'] = load_data(directory[1])
        # Merge the small components
        files[directory[0]]['Big small data'] = files[directory[0]]['Small data'][0]
        for i in range(1, len(files[directory[0]]['Small data'])):
            files[directory[0]]['Big small data'] = pd.merge(left=files[directory[0]]['Big small data'],
                                                             right=files[directory[0]]['Small data'][i],
                                                             left_index=True, right_index=True)
        files[directory[0]]['df_full'] = pd.merge(left=files[directory[0]]['Big small data'],
                                                  right=files[directory[0]]['Big data'],
                                                  left_index=True, right_index=True)
        files[directory[0]]['Correlation'], _ = correlation_with_cutoff(files[directory[0]]['df_full'], 0.8)
        files[directory[0]]['Half_Correlation'] = calculate_correlation(files[directory[0]]['Big small data'],
                                                                        files[directory[0]]['Big data'],
                                                                        files[directory[0]]['Correlation'])
    check_citrus(files)
    get_credibility(files)
    #consensus_big(files)
    #check_distribution(files)
