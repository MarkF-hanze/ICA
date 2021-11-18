import pandas as pd
import os
from scipy.stats import pearsonr
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from collections import defaultdict
from bokeh.plotting import figure, show, ColumnDataSource, output_file
from bokeh.palettes import Blues256, Set1, Category20
from bokeh.io import export_png
import holoviews as hv
from holoviews import opts
import dcor
from tqdm import tqdm
import random
import itertools
from sklearn.preprocessing import MinMaxScaler

hv.extension('bokeh')
pd.set_option('display.max_columns', None)


# TODO what to do when theer are 2 components
# TODO laatste check met cutoff 0.5


def load_data(load_directory,
              big_path='/home/MarkF/DivideConquer/Results/MathExperiment/2_Split/One_Normalized/ICARUN_ALL/'
                       'ica_independent_components_consensus.tsv'):
    small_df = []
    group_columns = {}
    for entry in os.scandir(load_directory):
        if 'ICARUN_SPLIT' in entry.path:
            for file in os.scandir(entry):
                if 'ica_independent_components_consensus.tsv' in file.path:
                    i = file.path.split('/')[-2][-1]
                    df = pd.read_csv(file.path, sep='\t', index_col=0)
                    print(f"Number of components split {i}: {df.shape[1]}")
                    df.columns = [f'{x}_s{i}' for x in df.columns]
                    group_columns[f's{i}'] = list(df.columns)
                    small_df.append(df)
    all_data = pd.read_csv(
        big_path, sep='\t', index_col=0)
    print(f'Number of components All data: {all_data.shape[1]}')
    all_data.columns = [f'{x}_big' for x in all_data.columns]
    group_columns[f'big'] = list(all_data.columns)
    return small_df, all_data, group_columns


def calculate_correlation(df1, df2):
    correlation = pd.DataFrame(columns=df1.columns, index=df2.columns)
    for column1 in df1:
        correlation[column1] = correlation[column1].astype(float)
        for column2 in df2:
            correlation.loc[column2, column1] = pearsonr(df1[column1].values, df2[column2].values)[0]
    correlation = correlation.abs()
    return correlation


def check_distribution(df):
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    ax3 = fig.add_subplot(gs[0, :])
    ax4 = fig.add_subplot(gs[1, :])
    ax1 = fig.add_subplot(gs[2:, 0])
    ax2 = fig.add_subplot(gs[2, 1])
    values = df.values.ravel()
    ax3.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 0.9, 1])
    ax2.set_title('Distribution pearson correlation')
    ax1.set_title('Log Distribution pearson correlation')
    sns.histplot(x=values, log_scale=False, kde=False, color='red', stat='density', ax=ax2)
    sns.histplot(x=values, log_scale=True, kde=True, color='red', stat='density', ax=ax1)
    sns.boxplot(x=values, ax=ax3)
    xs = np.linspace(0.2, 0.9, 8)
    ys = []
    for x in xs:
        ys.append((values >= x).sum())
    print(ys)
    ax4.plot(xs, ys, marker='o')
    ax4.set_xlabel("Pearson's cutoff")
    ax4.set_ylabel('Consensus count')
    ax4.grid(axis='both')
    # ax3.set_xlabel("Pearson's cutoff")
    ax3.get_shared_x_axes().join(ax3, ax4)
    plt.show()
    plt.clf()


def correlation_with_cutoff(df, stopping_point):
    correlation = df.corr(method='pearson')
    correlation = correlation.abs()
    correlation_cut_off = correlation.copy()
    correlation_cut_off[correlation_cut_off < stopping_point] = 0
    correlation_cut_off[correlation_cut_off >= stopping_point] = 1
    return correlation, correlation_cut_off


def make_clusters(df):
    df['Count'] = df.sum(axis=1)
    df = df.sort_values('Count', axis=0, ascending=False)
    temp_df = df.copy()
    local_cluster = []
    for index, row in df.iterrows():
        temp_df['Count'] = temp_df.sum(axis=0)
        if temp_df.loc[index, 'Count'] > 1:
            values = list(row[row > 0].index)
            # Remove count
            values = values[:-1]
            local_cluster.append(values)
            temp_df[values] = 0
    return local_cluster


def plot_global_heatmap(df):
    # Reorder and make global heatmap
    Z = linkage(df, method='ward', optimal_ordering=True)
    correlation = df.iloc[leaves_list(Z), leaves_list(Z)]
    palette = ['#ffffff', '#FF69B4']
    x_as = []
    y_as = []
    colors = []
    for x in correlation:
        for y in correlation:
            x_as.append(x)
            y_as.append(y)
            colors.append(palette[int(correlation.loc[x, y])])
    plot_df = pd.DataFrame()
    plot_df["x"] = x_as[::-1]
    plot_df["y"] = y_as[::-1]
    plot_df["colors"] = colors[::-1]
    tooltips = [
        ("x", "@x"),
        ("y", "@y"),
    ]
    p = figure(title="Categorical Heatmap", tooltips=tooltips,
               x_range=plot_df["x"].unique(), y_range=plot_df["y"].unique())
    source = ColumnDataSource(plot_df)
    p.rect("x", "y", color="colors", width=1, height=1, line_width=0.4, line_color="white", source=source, )
    p.axis.visible = False
    show(p)


def plot_local_heatmap(df):
    items = []
    for x in df:
        for y in df:
            items.append((x, y, df.loc[x, y]))
    hm = hv.HeatMap(items)
    hm.opts(opts.HeatMap(colorbar=True, width=800, height=800, tools=['hover'], xrotation=90, clim=(0, 1)))
    show(hv.render(hm))


def check_value(df, string, cluster):
    try:
        comp = [s for s in cluster if string in s][0]
        value = df[comp]
    except IndexError:
        value = np.zeros((df.shape[0]))
    return value


def make_negative(vector):
    return -vector


def do_nothing(vector):
    return vector


def merge_clusters(df, clusters):
    xs = []
    ys = []
    # Get the amount of small sets in df
    count = [z[1] for z in df.columns.str.split('_')]
    count = len(set(count)) - 1
    str_values = [f'_s{z}' for z in range(1, count + 1)]

    # Loop over every cluster
    for cluster in tqdm(clusters):
        # If the cluster has more than 2 values so a adding can happen
        if len(cluster) > 2:
            # Get the component for every set
            small_components = []
            for str_value in str_values:
                small_components.append(check_value(df, str_value, cluster))
            # Check or every small component the distance to the big component
            max_original_distances = 0
            if len([s for s in cluster if '_big' in s]) > 0:
                big_comp = df[[s for s in cluster if '_big' in s][0]]
                for component in small_components:
                    original_distance = dcor.distance_correlation(component, big_comp)
                    # If the correlation is higher make it the best components
                    if original_distance > max_original_distances:
                        max_original_distances = original_distance
                # Check every linear combination of the small components
                max_new_distances = 0
                # Make every combination of adding or subtracting
                operations = (do_nothing, make_negative)
                operations = itertools.product(operations, repeat=len(small_components))
                for operation in operations:
                    # Add all small components together (some are made negative so this is subtracting)
                    outcome = np.zeros((df.shape[0]))
                    for component, op in zip(small_components, operation):
                        outcome = outcome + op(component)

                    # Calculate the distance of this components to the big component
                    new_distances = dcor.distance_correlation(outcome, big_comp)
                    # If it is bigger add it
                    if new_distances > max_new_distances:
                        max_new_distances = new_distances
                xs.append(max_original_distances)
                ys.append(max_new_distances)
    # Make the scatter plot
    plt.scatter(xs, ys, marker="+", color="blue")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Original distance correlation')
    plt.ylabel('New distance correlation')
    plt.show()


def create_fake_lines(all_fakes, lines):
    for z, (name, row) in enumerate(all_fakes.iterrows()):
        try:
            lines = lines.append(pd.DataFrame([[row['Components'], all_fakes.iloc[z + 1, 1], 1, 0.0, 0, 0.0]],
                                              columns=list(lines.columns)), ignore_index=True)
        except IndexError:
            lines = lines.append(pd.DataFrame([[row['Components'], all_fakes.iloc[0, 1], 1, 0.0, 0, 0.0]],
                                              columns=list(lines.columns)), ignore_index=True)
    return lines


def create_fakes(nodes, lines, fake_count):
    new_df = pd.DataFrame()
    all_fakes = pd.DataFrame()
    z = 0
    for name, df in nodes.groupby('Group'):
        fake_df = pd.DataFrame()
        comp_count = fake_count
        fake_df['Group'] = np.repeat(f'fake{z}', comp_count)
        vector = ['consensus independent component ' + str(x) + f'_fake{z}' for x in range(0, comp_count)]
        fake_df['Components'] = vector
        vector = list(np.repeat('', df.shape[0]))
        fake_df['text'] = ''
        vector[df.shape[0] // 2] = name
        df['text'] = vector
        new_df = pd.concat([new_df, df, fake_df])
        all_fakes = pd.concat([fake_df, all_fakes])
        z += 1
    z = 0
    color_mapper = {}
    unique_count = len(nodes.Group.unique())
    for group in new_df.Group.unique():
        if 'fake' in group:
            color_mapper[group] = '#FFFFFF'
        else:
            if unique_count <= 9:
                color_mapper[group] = Set1[9][z]
            else:
                color_mapper[group] = Category20[20][z]
            z += 1
    new_df['node_color'] = [color_mapper[z] for z in new_df.Group]
    lines = create_fake_lines(all_fakes, lines)
    return new_df, lines


def citrus_plot(correlation):
    output_file(filename=f"citrusPlot.html")
    # Remove diagonal and values smaller than cutoff
    correlation.values[[np.arange(correlation.shape[0])] * 2] = 0
    #correlation[correlation < 0.3] = np.nan
    # Melt the dataframe to 3 columns
    lines = correlation.reset_index().melt(id_vars='index').dropna()
    summed = lines.copy()
    summed['value'] = [0 if z < 0.8 else z for z in summed['value']]
    summed = summed.groupby('index').sum()
    summed1 = lines.copy()
    summed1['value'] = [0 if z >= 0.8 else z for z in summed1['value']]
    summed1 = summed1.groupby('index').sum()
    # Drop the duplicates that like ab ba because correlation was mirrored
    lines = lines.loc[
        pd.DataFrame(np.sort(lines[['index', 'variable']], 1), index=lines.index).drop_duplicates(keep='first').index]
    lines['color'] = [0 if z < 0.8 else z for z in lines['value']]
    lines['alpha'] = [z for z in lines['value']]
    lines['alpha'] = MinMaxScaler(feature_range=(0, 1)).fit_transform(lines['alpha'].values.reshape(-1, 1))
    lines['width'] = [0.5 if z < 0.8 else 5 for z in lines['value']]
    lines['value'] = 1
    lines['value'] = lines['value'].astype(int)
    color_pallet = list(Blues256)[::-1]
    color_pallet[0] = '#D0D0D0'
    # Create a node for every component
    nodes = pd.DataFrame()
    nodes['Components'] = correlation.columns
    nodes['Group'] = [z.split('_')[1] for z in correlation.columns]
    nodes = pd.merge(left=nodes, right=summed, left_on='Components', right_index=True)
    nodes = pd.merge(left=nodes, right=summed1, left_on='Components', right_index=True)
    nodes = nodes.sort_values(['Group', 'value_x', 'value_y'], ascending=True)
    nodes = nodes.drop('value_x', axis=1)
    nodes = nodes.drop('value_y', axis=1)
    nodes, lines = create_fakes(nodes, lines, 15000)

    lines = lines.sort_values('color')
    # Make it holoview objects
    nodes = hv.Dataset(nodes, 'Components', ['Group', 'node_color', 'text'])
    # chord = hv.Chord((lines, nodes), ['index', 'variable'], ['value'])
    chord = hv.Chord((lines, nodes))
    chord.opts(
        opts.Chord(edge_color='color', edge_cmap=color_pallet, edge_alpha='alpha',
                   height=700, labels='text', node_color='node_color', label_text_color='node_color',
                   width=700, colorbar=True, edge_line_width='width', node_marker='none', node_radius=5,
                   label_text_font_size='40px', colorbar_position='top',
                   colorbar_opts={'width': 500, 'title': 'Pearson correlation'}),
    )
    # chord.opts(toolbar=None, default_tools = []
    #     opts.Chord(edge_color='color', edge_cmap=color_pallet, edge_alpha='alpha',
    #                labels='text', node_color='node_color',
    #                fig_size=(15, 15),
    #                colorbar=True,  edge_linewidth='width', node_marker='none',
    #                text_font_size='40px',
    #                colorbar_opts={'width': 500, 'title': 'Pearson correlation'}))
    chord = chord.redim.range(color=(0, 1))
    hv.save(chord, 'citrusPlot.png', fmt='png')
    #chord = hv.render(chord, backend='bokeh')
    #chord.toolbar.autohide = True
    #export_png(chord, filename="citrusPlot.png")


def get_credibility(clusters, directory):
    # Load the different credibility scores
    small_df = []
    sets = []
    for entry in os.scandir(directory):
        if 'ICARUN_SPLIT' in entry.path:
            for file in os.scandir(entry):
                if 'ica_robustness_metrics_independent_components_consensus.tsv' in file.path:
                    z = file.path.split('/')[-2][-1]
                    df = pd.read_csv(file.path, sep='\t', index_col=0)
                    df.index = [f'{q}_s{z}' for q in df.index]
                    small_df.append(df)
                    sets.append(f's{z}')
    # Change index
    all_data = pd.read_csv(
        '/home/MarkF/DivideConquer/Results/MathExperiment/2_Split/One_Normalized/ICARUN_ALL/'
        'ica_robustness_metrics_independent_components_consensus.tsv', sep='\t', index_col=0)
    all_data.index = [f'{z}_big' for z in all_data.index]
    sets.append(f'big')
    small_df.append(all_data)
    df_big = pd.concat(small_df)
    # Get the distribution of all the components and the credibility index
    all_values = {}
    for s, df in zip(sets, small_df):
        all_values[s] = df[f'credibility index']

    # Get the distribution of only the variables that have consensus
    consensus_values = defaultdict(list)
    for cluster in clusters:
        for value in cluster:
            consensus_values[value.split('_')[1]].append(df_big.loc[value, 'credibility index'])
    # Make the histogram
    for value in consensus_values:
        sns.histplot(consensus_values[value], kde=True, color='red',
                     label='Consensus variables')
        sns.histplot(all_values[value].values, kde=True, color='blue', label='All variables')
        plt.legend()
        plt.title(value)
        #plt.show()
    return all_values


# Todo add cutoff
def big_vs_small(correlation, groups):
    big_correlation = correlation.copy()
    big_correlation[big_correlation < 0.65] = np.nan
    big_component = big_correlation.count(axis=1)
    big_component = big_component[big_component > 2]
    data = {}
    end_results = {}
    components = {}
    for component in tqdm(set(groups["big"]).intersection(set(big_component.index))):
        df = big_correlation.loc[component].dropna().drop(component)
        if component == 'consensus independent component 155_big':
            print(df)
        data[component] = [df.mean()]
        temp = list(df.index)
        temp.append(component)
        components[component] = temp
        combinations = itertools.combinations(df.index, 2)
        cor_values = []
        for combination in combinations:
            cor_values.append(correlation.loc[combination[0], combination[1]])
        data[component].append(np.mean(cor_values))
        end_results[component] = data[component][0] - data[component][1]
    high_to_low = [k for k, v in sorted(end_results.items(), key=lambda item: item[1])][::-1]
    fig, axs = plt.subplots(int(np.round(np.sqrt(len(high_to_low)))), int(round(np.sqrt(len(high_to_low))))
                            , sharey=True)
    for z in range(len(high_to_low)):
        df = correlation.loc[components[high_to_low[z]], components[high_to_low[z]]]
        df = df.reset_index().melt(id_vars='index')
        df = df.loc[
            pd.DataFrame(np.sort(df[['index', 'variable']], 1), index=df.index).drop_duplicates(keep='first').index]
        df = df[df['index'] != df['variable']]
        df['name'] = df['index'].str.split('_', expand=True)[1] + ' vs ' + df['variable'].str.split('_', expand=True)[1]
        df = df.sort_values(['name'], axis=0)
        colors = [[], []]
        cm = plt.get_cmap('Dark2')

        for name in df['name']:
            splitter = name.split(' vs ')
            for q, split in enumerate(splitter):
                if 'big' in split:
                    colors[q].append('black')
                elif 's1' in split:
                    colors[q].append(cm(0))
                elif 's2' in split:
                    colors[q].append(cm(1))
                elif 's3' in split:
                    colors[q].append(cm(2))
                elif 's4' in split:
                    colors[q].append(cm(3))

        axs.ravel()[z].axes.get_xaxis().set_ticks([])
        axs.ravel()[z].bar(df['name'], (df['value'] / 10) * 9, color=colors[0])
        axs.ravel()[z].bar(df['name'], (df['value'] / 10) * 1, bottom=(df['value'] / 10) * 9, color=colors[1])
    plt.tight_layout()
    plt.show()


def consensus_small(df, groups, credibility_dict):
    df = df.drop(groups['big'], axis=1)
    df_dcor = df.corr(method=dcor.distance_correlation)
    #df_dcor = df.corr()
    df_dcor_original = df_dcor.copy()
    #plot_local_heatmap(df_dcor)
    df_dcor = df_dcor[df_dcor > 0.5]
    consensus_df = df_dcor.count()
    consensus_df = consensus_df[consensus_df > 2].sort_values(ascending=False)
    estimated_sources = list(consensus_df.index)
    credibility_df = pd.DataFrame()
    for group in credibility_dict:
        if group != 'big':
            credibility_df = pd.concat([credibility_df, credibility_dict[group]], ignore_index=False, axis=0)
            #credibility_df = credibility_df.append(credibility_dict[group])
    drop_columns = []
    while len(estimated_sources) > 0:
        estimated_source_df = df_dcor[estimated_sources[0]].dropna()
        #print(estimated_source_df)
        estimated_sources = [z for z in estimated_sources if z not in estimated_source_df.index]
        #print(credibility_df.loc[estimated_source_df.index,:])
        drop_columns.extend(list(credibility_df.loc[estimated_source_df.index, :].sort_values(
            by=0, ascending=False).iloc[1:].index))
    df_dcor_original = df_dcor_original.drop(set(drop_columns), axis=0)
    df_dcor_original = df_dcor_original.drop(set(drop_columns), axis=1)
    print(df_dcor_original)



if __name__ == "__main__":
    # Load the small and big data
    directory = '/home/MarkF/DivideConquer/Results/Math_Clustered/4_Split/'
    # directory = '/home/MarkF/DivideConquer/Results/MathExperiment/2_Split/One_Normalized'
    # directory = '/home/MarkF/DivideConquer/Results/MathExperiment/3_Split/'
    small_data, bigdata, lookup_columns = load_data(directory)
    # Create the fake clusters for the check later
    fake_clusters = []
    for x in range(50):
        fake_cluster = []
        for data in small_data:
            fake_cluster.append(random.choice(list(data.columns)))
        fake_cluster.append(random.choice(list(bigdata.columns)))
        fake_clusters.append(fake_cluster)
    # Merge the small components
    big_small_data = small_data[0]
    for i in range(1, len(small_data)):
        big_small_data = pd.merge(left=big_small_data, right=small_data[i], left_index=True, right_index=True)
    # Only compare these groups for now
    compare_groups = [(big_small_data, bigdata)]
    # Pearson cutoff for consensus
    cut_off = 0.8
    for dataframe_group in compare_groups:
        # Check how the correlation is distributed
        # half_correlation = calculate_correlation(dataframe_group[0], dataframe_group[1])
        # check_distribution(half_correlation)
        # Merge everything together
        df_full = pd.merge(left=dataframe_group[0], right=dataframe_group[1], left_index=True, right_index=True)
        # Get correlation and make only 0,1 based on cutoff
        full_correlation, full_correlation_cut_off = correlation_with_cutoff(df_full, cut_off)
        # big_vs_small(full_correlation, lookup_columns)
        # plot_global_heatmap(full_correlation_cut_off)
        # Citrus plot of how the variables are correlated
        citrus_plot(full_correlation)
        # Make the clusters based on the correlation
        clusters = make_clusters(full_correlation_cut_off)
        # See how the credibility is distributed
        credibility = get_credibility(clusters, directory)
        consensus_small(df_full, lookup_columns, credibility)
        # sys.exit()
        # most_repr_sets = get_representative_groups(df_full, clusters)
        # See if the correlation gets better when merging clusters
        # merge_clusters(df_full, clusters)
        # merge_clusters(df_full, fake_clusters)
