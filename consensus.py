import pandas as pd
import numpy as np
import os
import sys

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from bokeh.plotting import figure, ColumnDataSource
from bokeh.palettes import Blues256, Category20
import holoviews as hv
from holoviews import opts
from bokeh.io import export_png, export_svgs

from scipy.cluster.hierarchy import linkage, leaves_list
import dcor

from collections import defaultdict
from tqdm import tqdm
import random
import itertools

from Correlation import Correlation
from LoadData import LoadICARuns
from BiologicalInterpertation import MergeTwo

hv.extension('bokeh')
pd.set_option('display.max_columns', None)


def load_big(path):
    """
    Helper function to load the big data file and rename the columns to fit with the other dataframes.
    Every column should be splitted with the name as underbracket
     """
    all_data = pd.read_csv(
        path, sep='\t', index_col=0)
    print(f'Number of components All data: {all_data.shape[1]}')
    # Make the columns unique so they later can be added to the same dataframe
    all_data.columns = [f'{x}_big' for x in all_data.columns]
    return all_data


def load_data(load_directory,
              big_path='/home/MarkF/DivideConquer/Results/MathExperiment/2_Split/One_Normalized/ICARUN_ALL/'
                       'ica_independent_components_consensus.tsv'):
    """
    Load the math experiment data. This function loads ether the clustered or the random split.
    load_directory is the folder that contains the split
    big_path contains the file with all components
     """
    # List to save the small splits
    small_df = []
    # List to save the name of each split
    group_columns = {}
    # Loop over the directory
    for entry in os.scandir(load_directory):
        # Load the information of the seperate splits these clusters should be called 'ICARUN_SPLIT'
        if 'ICARUN_SPLIT' in entry.path:
            # Loop over all the files and find the independent component consensus
            # TODO this doesnt have to be in a for loop
            for file in os.scandir(entry):
                if 'ica_independent_components_consensus.tsv' in file.path:
                    # Get the split name
                    i = file.path.split('/')[-2][-1]
                    # Load the component
                    df = pd.read_csv(file.path, sep='\t', index_col=0)
                    # Print it to check
                    print(f"Number of components split {i}: {df.shape[1]}")
                    # Set the names as the split type
                    # TODO REMOVE THE s
                    df.columns = [f'{x}_s{i}' for x in df.columns]
                    # Add id and dataframe to the relative listst
                    group_columns[f's{i}'] = list(df.columns)
                    small_df.append(df)
    # Load the big dataset
    all_data = load_big(big_path)
    group_columns[f'big'] = list(all_data.columns)
    return small_df, all_data, group_columns


# Specific function to load cancer type information because of different formatting
# only used for paper can be removed in final build
def load_cancer_type(load_directory,
                     big_path='/home/MarkF/DivideConquer/Results/GPL570/All_Cancer/ICARUN/'
                              'ica_independent_components_consensus.tsv'):
    """
    Load the cancer ICA data.
    load_directory is the folder that contains the split
    big_path contains the file with all components
     """
    # List to save the small splits
    small_df = []
    # List to save the name of each split
    group_columns = {}
    # Loop over the cancer type folder
    for cancer_type in os.scandir(load_directory):
        # For this specific case do not load all cancer data
        if 'All_Cancer' not in cancer_type.path:
            # Also do not load any files that may be present
            if os.path.isdir(cancer_type):
                # TODO this doesnt have to be in a for loop
                # Search th consensus file
                for file in os.scandir(f'{cancer_type.path}/ICARUN'):
                    if 'ica_independent_components_consensus.tsv' in file.path:
                        # Get the name of the cancer type
                        i = cancer_type.path.split('/')[-1].replace('_', ' ')
                        # Load the file
                        df = pd.read_csv(file.path, sep='\t', index_col=0)
                        # Print for the check
                        print(f"Number of components split {i}: {df.shape[1]}")
                        # Give the column a unique name and add the relative values to the correct list
                        df.columns = [f'{x}_{i}' for x in df.columns]
                        group_columns[f'{i}'] = list(df.columns)
                        small_df.append(df)
    # Load the big dataset
    all_data = load_big(big_path)
    group_columns[f'big'] = list(all_data.columns)
    return small_df, all_data, group_columns

def correlation_with_cutoff(df, stopping_point):
    correlation = np.corrcoef(df.values, rowvar=False)
    correlation = np.absolute(correlation)
    correlation = pd.DataFrame(correlation, columns=df.columns, index=df.columns)
    correlation_cut_off = correlation.copy()
    correlation_cut_off[correlation_cut_off < stopping_point] = 0
    correlation_cut_off[correlation_cut_off >= stopping_point] = 1
    return correlation, correlation_cut_off


def make_clusters(df):
    """
    Insert correlation matrix with only 0 and 1 with a certain cutoff
    """
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


def plot_global_heatmap(df, name):
    # Reorder and make global heatmap
    palette = ['#0000FF', '#00FFFF']
    x_as = []
    y_as = []
    colors = []
    for y in df.index:
        for x in df.columns:
            y_as.append(y)
            x_as.append(x)
            colors.append(palette[int(df.loc[y, x])])
    plot_df = pd.DataFrame()
    plot_df["x"] = x_as
    plot_df["y"] = y_as
    plot_df["colors"] = colors
    tooltips = [
        ("x", "@x"),
        ("y", "@y"),
    ]
    p = figure(title="Categorical Heatmap", tooltips=tooltips,
               x_range=plot_df["x"].unique(), y_range=plot_df["y"].unique())
    source = ColumnDataSource(plot_df)
    p.rect("x", "y", color="colors", width=1, height=1, line_width=0.4, line_color="white", source=source, )
    p.axis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None
    # p.output_backend = "svg"
    # export_svgs(p, filename=f'{save_directory}/Global_Correlation.svg')
    export_png(p, filename=f'{save_directory}/{name}.png')


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


def merge_clusters(df, clusters, name):
    xs = []
    ys = []
    # Get the amount of small sets in df
    groups = [z[1] for z in df.columns.str.split('_')]
    groups = set(groups)
    groups = [f'_{z}' for z in groups if z != 'big']
    # Loop over every cluster
    for cluster in tqdm(clusters):
        if len(set([x.split('_')[1] for x in cluster])) != len(cluster):
            continue
        # If the cluster has more than 2 values so a adding can happen
        if len(cluster) > 2:
            # Get the component for every set
            small_components = []
            for group in groups:
                small_components.append(check_value(df, group, cluster))
            # Check for every small component the distance to the big component
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
    plt.clf()
    plt.scatter(xs, ys, marker="+", color="blue")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Original distance correlation')
    plt.ylabel('New distance correlation')
    plt.savefig(f'{save_directory}/Fading_{name}.svg', dpi=1200)
    plt.clf()


def create_fake_lines(all_fakes, lines):
    for z, (name, row) in enumerate(all_fakes.iterrows()):
        try:
            lines = lines.append(pd.DataFrame([[row['Components'], all_fakes.iloc[z + 1, 1], 1, 0.0, 0, 1]],
                                              columns=list(lines.columns)), ignore_index=True)
        except IndexError:
            lines = lines.append(pd.DataFrame([[row['Components'], all_fakes.iloc[0, 1], 1, 0.0, 0, 1]],
                                              columns=list(lines.columns)), ignore_index=True)
    return lines


def create_fakes(nodes, lines, fake_count, colors_mapped):
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
        print(name)
        vector[df.shape[0] // 2] = name
        # vector[df.shape[0] // 2] = ''
        df['text'] = vector
        new_df = pd.concat([new_df, df, fake_df])
        all_fakes = pd.concat([fake_df, all_fakes])
        z += 1
    z = len(colors_mapped)
    unique_count = len(nodes.Group.unique())
    for group in new_df.Group.unique():
        if group not in colors_mapped:
            if 'fake' in group:
                colors_mapped[group] = '#FFFFFF'
            else:
                colors_mapped[group] = Category20[20][z]
                z += 1
    print(colors_mapped)
    new_df['node_color'] = [colors_mapped[z] for z in new_df.Group]
    lines = create_fake_lines(all_fakes, lines)
    return new_df, lines


def get_sum(lines, bigger):
    summed = lines.copy()
    if bigger:
        summed['alpha'] = [0 if z < 0.6 else z for z in summed['alpha']]
    else:
        summed['alpha'] = [0 if z >= 0.6 else z for z in summed['alpha']]
    total_value = defaultdict(lambda: 0)
    for index, row in summed.iterrows():
        total_value[row['index']] = total_value[row['index']] + row['alpha']
        total_value[row['variable']] = total_value[row['variable']] + row['alpha']
    summed = pd.DataFrame(list(total_value.items()), columns=['Components', 'value'])
    return summed


def citrus_plot(correlation, colors_mapped):
    # output_file(filename=f"{save_directory}/citrusPlot.html")
    # Remove diagonal and values smaller than cutoff
    correlation.values[[np.arange(correlation.shape[0])] * 2] = np.nan
    # Melt the dataframe to 3 columns
    lines = correlation.reset_index().melt(id_vars='index').dropna()
    # Drop the duplicates that like ab ba because correlation was mirrored
    lines = lines.loc[
        pd.DataFrame(np.sort(lines[['index', 'variable']], 1), index=lines.index).drop_duplicates(keep='first').index]
    lines = lines[lines['index'].str.split('_', expand=True)[1] != lines['variable'].str.split('_', expand=True)[1]]
    start_nodes = set(lines[['index', 'variable']].values.ravel())
    lines = lines[lines['value'] > 0.05]
    end_nodes = set(lines[['index', 'variable']].values.ravel())
    missing_df = pd.DataFrame(start_nodes.difference(end_nodes), columns=['Components'])
    missing_df['Groups'] = missing_df['Components']
    # lines['color'] = [0 if z < 0.8 else z for z in lines['value']]
    lines['color'] = [z for z in lines['value']]
    lines['alpha'] = [z for z in lines['value']]
    # Runs compare
    lines['width'] = 1
    # Math experiment
    # lines['width'] = [.3 if z < 0.6 else .9 for z in lines['value']]
    # Cancer settings
    # lines['width'] = [.08 if z < 0.6 else .9 for z in lines['value']]
    # lines['width'] = .3
    lines['value'] = 1
    lines['value'] = lines['value'].astype(int)
    # Make until 0.6 grey
    color_pallet = np.array(list(Blues256)[::-1])
    size = round((len(color_pallet) / 4) * 6)
    greys = np.repeat('#D0D0D0', size).tolist()
    greys.extend(list(color_pallet))
    # color_pallet[0: round((len(color_pallet) / 10) * 6)] = '#D0D0D0'
    color_pallet = greys
    # Add the missing lines back due to not plotting certain lines
    lines = create_fake_lines(missing_df, lines)
    # Create a node for every component
    nodes = pd.DataFrame()
    nodes['Components'] = correlation.columns
    nodes['Group'] = [z.split('_')[1] for z in correlation.columns]
    # Do the sorting
    summed = get_sum(lines, True)
    summed1 = get_sum(lines, False)

    nodes = pd.merge(left=nodes, right=summed, left_on='Components', right_on='Components')
    nodes = pd.merge(left=nodes, right=summed1, left_on='Components', right_on='Components')
    nodes = nodes.sort_values(['Group', 'value_x', 'value_y'], ascending=True)
    nodes = nodes.drop('value_x', axis=1)
    nodes = nodes.drop('value_y', axis=1)

    nodes, lines = create_fakes(nodes, lines, 1000, colors_mapped)

    lines = lines.sort_values('color')
    # Make it holoview objects
    nodes = hv.Dataset(nodes, 'Components', ['Group', 'node_color', 'text'])
    # chord = hv.Chord((lines, nodes), ['index', 'variable'], ['value'])
    chord = hv.Chord((lines, nodes))
    # '40px'
    chord.opts(
        opts.Chord(edge_color='color', edge_cmap=color_pallet, edge_alpha='alpha',
                   height=2000, labels='text', node_color='node_color', label_text_color='node_color',
                   width=2000, colorbar=True, edge_line_width='width', node_marker='none', node_radius=5,
                   label_text_font_size='100px', colorbar_position='top',
                   colorbar_opts={'width': 500, 'title': 'Pearson correlation'}),
    )
    chord = chord.redim.range(color=(0, 1))
    hv.save(chord, f'{save_directory}/citrusPlot.png')
    plot_state = hv.renderer('bokeh').get_plot(chord).state
    plot_state.output_backend = 'svg'
    #export_svgs(plot_state, filename=f'{save_directory}/citrusPlot.svg')





def biology_small(correlation):
    df = correlation.reset_index().melt(id_vars='index').dropna()
    df = df[df['index'].str.split('_', expand=True)[1] != df['variable'].str.split('_', expand=True)[1]]
    df['vs group'] = df['variable'].str.split('_', expand=True)[1].values
    df = df[~df['index'].str.contains('big')]
    #df = df[df.groupby(['index', 'vs group'])['value'].transform(max) == df['value']]
    print(df)
    sys.exit()
    best_scores = {
        'Small Component': [],
        'Big Component': [],
        'Small2 Component': [],
        'Big2 Component': [],
        'Score': [],
    }
    for component, comp_df in tqdm(df.groupby('index')):
        big_cor = comp_df[comp_df['vs group'] == 'big']['value'].values[0]
        big_component = comp_df[comp_df['vs group'] == 'big']['variable'].values[0]
        comp_df = comp_df[comp_df['vs group'] != 'big']
        # Get the correlation of the other components with big
        df2 = df[df['index'].isin(comp_df['variable'])]
        df2 = df2[df2['vs group'] == 'big']
        comp_df = pd.merge(left=comp_df, right=df2, left_on='variable', right_on='index')
        a = np.array(comp_df['value_y'].values.tolist())
        comp_df['value_y'] = np.where(a < big_cor, big_cor, a).tolist()
        comp_df['Score'] = comp_df['value_x'] - comp_df['value_y']
        comp_df = comp_df.sort_values(by='Score', ascending=False)
        best_scores['Small Component'].append(component)
        best_scores['Big Component'].append(big_component)
        best_scores['Small2 Component'].append(comp_df.iloc[0, :]['variable_x'])
        best_scores['Big2 Component'].append(comp_df.iloc[0, :]['variable_y'])
        best_scores['Score'].append(comp_df.iloc[0, :]['Score'])
    df = pd.DataFrame.from_dict(best_scores, orient='index').transpose().sort_values(by='Score', ascending=False)
    # Remove the same values
    df = df.loc[
        pd.DataFrame(np.sort(df[['Small Component', 'Small2 Component']], 1), index=df.index).drop_duplicates(
            keep='first').index]
    df.to_csv(f'{save_directory}/Biological_int.csv')
    # Bigger than 0.5 count it
    df = df[df['Score'] > 0.5]
    color_mapper = plot_histogram(correlation, df.drop('Score', axis=1).iloc[:].values, "Biological_int")
    return color_mapper


def plot_histogram(correlation, columns, name_file):
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
        df['name'] = df['index'].str.split('_', expand=True)[1] + ' vs ' + df['variable'].str.split('_', expand=True)[1]
        df = df.sort_values(['name'], axis=0)
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
    plt.savefig(f'{save_directory}/{name_file}.svg', dpi=1200,
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    return color_mapper


# TODO gaat dit goed??
def big_vs_small(correlation):
    # Count how often a estimated source correlates with another source with cutoff 0.6
    big_correlation = correlation.copy()
    big_correlation[big_correlation < 0.6] = np.nan
    big_component = big_correlation.count(axis=1)
    # Dictionaries to save the results
    data = {}
    end_results = {}
    components = {}
    # Get the components that are both in the big group and have a correlation with at least 1 small component
    # Only leave components that correlate with at least 2 others (bigger than 2 because it also correlated with
    # itself)
    big_component = big_component[big_component > 2]
    all_big_components = [i for i in correlation.columns if "big" in i]
    loop = set(all_big_components).intersection(set(big_component.index))
    for component in tqdm(loop):
        # Get the big source and the highly correlated sources and drop the big component
        print(big_correlation.loc[component].dropna())
        with_duplicated_df = pd.DataFrame(big_correlation.loc[component].dropna().drop(component))
        with_duplicated_df["Group"] = with_duplicated_df.reset_index()["index"].str.split(
            "_", expand=True).iloc[:, 1].values
        # Remove correlation with itself
        with_duplicated_df = with_duplicated_df[with_duplicated_df["Group"] != component.split("_")[-1]]
        unique_groups = len(with_duplicated_df["Group"].unique())
        if unique_groups >= 2:
            groups = with_duplicated_df.groupby("Group").count()
            # Get all how often a single group appears
            groups = groups[groups[component] >= 2]
            if len(groups) > 0:
                all_combinations = []
                for name, group in with_duplicated_df.groupby("Group"):
                    all_combinations.append(list(group.index))
                unique_dfs = itertools.product(*all_combinations)
                unique_dfs = [list(z) for z in unique_dfs]
            else:
                unique_dfs = [list(with_duplicated_df.index)]
            for z, unique_index in enumerate(unique_dfs):
                df = with_duplicated_df.loc[unique_index, component]
                # Take the mean of the remaining component, this is the distance from big to the smaller components
                data[f"{component}_{z}"] = [df.mean()]
                # Take all components again and append them to see what components were used
                temp = list(df.index)
                temp.append(component)
                components[f"{component}_{z}"] = temp
                # Get all possible combinations of small components with each other
                combinations = itertools.combinations(df.index, 2)
                # Get the correlation of these combinations
                cor_values = []
                for combination in combinations:
                    cor_values.append(correlation.loc[combination[0], combination[1]])
                # Append the mean distance of all the small components between each other
                data[f"{component}_{z}"].append(np.mean(cor_values))
                # Best components have high correlation between big and small and low between small small
                score = data[f"{component}_{z}"][0] - data[f"{component}_{z}"][1]
                # if score > 0.2:
                end_results[f"{component}_{z}"] = score
    # Sort the end results to see the best components
    high_to_low = [k for k, v in sorted(end_results.items(), key=lambda item: item[1])][::-1]
    input_figure = [components[x] for x in high_to_low]
    pd.DataFrame(input_figure).to_csv(f'{save_directory}/EC_splitted.csv')
    plot_histogram(correlation, input_figure, "EC_splitted")


# TODO alles opnieuw runnen
# TODO testen
if __name__ == "__main__":
    # Load the small and big data
    #directory = '/home/MarkF/DivideConquer/Results/Math_Clustered/2_Split/'
    # directory = '/home/MarkF/DivideConquer/Results/MathExperiment/2_Split/One_Normalized'
    # directory = '/home/MarkF/DivideConquer/Results/MathExperiment/4_Split/'
    # small_data, bigdata, lookup_columns = load_data(directory,
    #                                                 '/home/MarkF/DivideConquer/Results/MathExperiment/2_Split/One_Normalized/ICARUN_ALL/'
    #                                                 'ica_independent_components_consensus.tsv')
    datasets = LoadICARuns('/home/MarkF/DivideConquer/Results/2000_Samples_Experiment/Clustered_vs_Random_Experiment/'
                           'Clustered_Splits/2_Split',
                            '/home/MarkF/DivideConquer/Results/2000_Samples_Experiment/Clustered_vs_Random_Experiment/'
                            'ICARUN_ALL/ica_independent_components_consensus.tsv')
    #small_data, bigdata, lookup_columns = load_cancer_type('/home/MarkF/DivideConquer/Results/GPL570')
    # small_data, bigdata, lookup_columns = load_cancer_type('/home/MarkF/DivideConquer/Results/MathBlood')
    save_directory = '/home/MarkF/DivideConquer/ICA/Results/Clustered/2_Split'
    # save_directory = '/home/MarkF/DivideConquer/ICA/Results/0_Cred'
    # Create the fake clusters for the check later
    fake_clusters = []
    for x in range(50):
        fake_cluster = []
        for split in datasets.get_individual_small():
            fake_cluster.append(random.choice(list(datasets.get_individual_small()[split].columns)))
        fake_cluster.append(random.choice(list(datasets.get_sample_data().columns)))
        fake_clusters.append(fake_cluster)

    # Pearson cutoff for consensus
    cut_off = 0.6
    # Merge everything together
    # Get correlation and make only 0,1 based on cutoff
    correlation = Correlation(datasets.get_merged_small(), datasets.get_sample_data())
    MergeTwo(correlation.get_correlation())
   # color_mapper = biology_small(correlation.get_correlation())

    # Turn it to later colors for the html plot
    for z in color_mapper:
        if z != 'big':
            rgb = tuple([int(x * 255) for x in color_mapper[z]])
            color_mapper[z] = '#%02x%02x%02x' % rgb[:3]
        else:
            color_mapper[z] = '#000000'
    citrus_plot(correlation.get_correlation(), color_mapper)
    big_vs_small(correlation.get_correlation())
    # Cluster and plot
    test_df = correlation.get_one_zero_correlation(cut_off).sum()
    test_df = test_df[test_df > 1]
    # Z = linkage(full_correlation_cut_off.values, method='ward', optimal_ordering=True)
    Z = linkage(correlation.get_one_zero_correlation(cut_off)[test_df.index].values, method='ward', optimal_ordering=True)
    cols = [correlation.get_one_zero_correlation(cut_off).columns[x] for x in leaves_list(Z)]
    for column in correlation.get_one_zero_correlation(cut_off):
        if column not in cols:
            cols.append(column)
    clustered_df = correlation.get_one_zero_correlation(cut_off).loc[cols, cols]
    plot_global_heatmap(clustered_df.iloc
                        [:100, :100]
                        , 'Left_Corner_Heatmap')
    # Make the clusters based on the correlation
    clusters = make_clusters(correlation.get_one_zero_correlation(cut_off))
    # See how the credibility is distributed
    # See if the correlation gets better when merging clusters
    merge_clusters(correlation.get_merged_normall(), clusters, 'clustered')
    merge_clusters(correlation.get_merged_normall(), fake_clusters, 'random')
