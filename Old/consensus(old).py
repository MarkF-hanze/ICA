import pandas as pd
from scipy.stats import pearsonr, spearmanr
import scipy.stats as st
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, fcluster
from scipy.spatial import distance
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models import Span
import holoviews as hv
from holoviews import opts
from tqdm import tqdm
hv.extension('bokeh')

# TODO I normalized it per dataset
# TODO werkt nu alleen voor False omdat ik alleen die zonder x_y check bij clusteren
COMPARE_CUTS = False
if COMPARE_CUTS:
    ces = pd.read_csv(
        '/home/MarkF/DivideConquer/Results/MathExperiment/Split1/ica_independent_components_consensus.tsv',
        sep='\t', index_col=0)
    print(f'Number of components split 1: {ces.shape[1]}')

    ces1 = pd.read_csv(
        '/home/MarkF/DivideConquer/Results/MathExperiment/Split2/ica_independent_components_consensus.tsv',
        sep='\t', index_col=0)
    print(f'Number of components split 2: {ces1.shape[1]}')
else:
    ces_cut1 = pd.read_csv(
        '/home/MarkF/DivideConquer/Results/MathExperiment/Split1/ica_independent_components_consensus.tsv',
        sep='\t', index_col=0)

    ces_cut2 = pd.read_csv(
        '/home/MarkF/DivideConquer/Results/MathExperiment/Split2/ica_independent_components_consensus.tsv',
        sep='\t', index_col=0)
    ces = pd.merge(left=ces_cut1, right=ces_cut2, left_index=True, right_index=True)
    print(f'Number of components split 1 and 2 combined (Without consensus): {ces.shape[1]}')
    ces1 = pd.read_csv('/home/MarkF/DivideConquer/Results/MathExperiment/All/ica_independent_components_consensus.tsv',
                       sep='\t', index_col=0)
    print(f'Number of components All data: {ces1.shape[1]}')

# Boxplot
correlation = pd.DataFrame(index=ces1.columns, columns=ces.columns)

for column in ces:
    correlation[column] = correlation[column].astype(float)
    for column1 in ces1:
        correlation.loc[column1, column] = pearsonr(ces[column].values, ces1[column1])[0]

correlation = correlation.abs()
values = correlation.abs().values.ravel()



# fig, (ax1, ax2, ax3) = plt.subplots(1, 3,)
ax3 = plt.subplot(2, 1, 1)
ax1 = plt.subplot(2, 2, 3)
ax2 = plt.subplot(2, 2, 4)
ax3.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 0.9, 1])
ax2.set_title('Distribution pearson correlation')
ax1.set_title('Log Distribution pearson correlation')
sns.histplot(x=values, log_scale=False, kde=False, color='red', stat='density', ax=ax2)
sns.histplot(x=values, log_scale=True, kde=True, color='red', stat='density', ax=ax1)
sns.boxplot(x=values, ax=ax3)



# Heatmap
cut_off_components = []
cut_offs = np.linspace(0.1, 0.9, 9)[::-1]
for cut_off in tqdm(cut_offs):
    correlation_start = pd.merge(left=ces, right=ces1, left_index=True, right_index=True)
    correlation = correlation_start.corr(method='pearson')
    # Only leave those above a threshold
    correlation = correlation.abs()
    correlation[correlation < cut_off] = 0
    correlation[correlation >= cut_off] = 1
    # Clustering
    clusters = []
    already_added = []
    for index, row in correlation.iterrows():
        if ('_x' not in index) and ('_y' not in index):
            positive_values = list(row[row > 0].index)
            if len(positive_values) > 1:
                if index not in already_added:
                    already_added.extend(positive_values)
                    clusters.append(positive_values)

    # Reorder and make global heatmap
    Z = linkage(correlation, method='ward', optimal_ordering=True)
    correlation = correlation.iloc[leaves_list(Z), leaves_list(Z)]
    palette = ['#ffffff', '#FF69B4']
    x_as = []
    y_as = []
    colors = []
    for x in correlation:
        for y in correlation:
            x_as.append(x)
            y_as.append(y)
            colors.append(palette[int(correlation.loc[x, y])])
    df = pd.DataFrame()
    df["x"] = x_as[::-1]
    df["y"] = y_as[::-1]
    df["colors"] = colors[::-1]
    TOOLTIPS = [
        ("x", "@x"),
        ("y", "@y"),
    ]
    p = figure(title="Categorical Heatmap", tooltips=TOOLTIPS,
               x_range=df["x"].unique(), y_range=df["y"].unique())
    source = ColumnDataSource(df)
    p.rect("x", "y", color="colors", width=1, height=1, line_width=0.4, line_color="white", source=source,)
    p.axis.visible = False
    if cut_off == 0.8:
        show(p)

    # Heatmap per group
    first_run = True
    most_repr_sets = []
    for cluster in clusters:
        if len(cluster) > 2:
            correlation = pd.DataFrame(index=cluster, columns=cluster)
            for column in cluster:
                for column1 in cluster:
                    # Higher than 1 is negative correlation so substract that
                    cor_value = distance.correlation(correlation_start[column], correlation_start[column1])
                    if cor_value > 1:
                        cor_value -= 1
                    correlation.loc[column1, column] = cor_value
                correlation[column] = correlation[column].astype(float)
            most_repr_sets.append(correlation.sum().idxmin())
            # Make the heatmap
            if first_run:
                items = []
                for x in correlation:
                    for y in correlation:
                        items.append((x, y, correlation.loc[x, y]))
                hm = hv.HeatMap(items)
                hm.opts(opts.HeatMap(colorbar=True, width=800, height=800, tools=['hover'], xrotation=90))
                if cut_off == 0.8:
                    show(hv.render(hm))
                first_run = False
        else:
            most_repr_sets.append(cluster[0])
    cut_off_components.append(len(most_repr_sets))
    if cut_off == 0.8:
        print(most_repr_sets)
p = figure(width=400, height=400, x_axis_label="Pearson's cutoff", y_axis_label='Count consensus components')
p.line(cut_offs, cut_off_components, line_width=2)
p.circle(cut_offs, cut_off_components, size=5)
# Horizontal line
hline = Span(location=min(ces.shape[1], ces1.shape[1]), dimension='width', line_color='red', line_width=2,
             line_dash='dashed')
p.renderers.extend([hline])
show(p)
plt.show()

def plot_local_heatmap(df, i):
    output_file(f'{save_directory}/Local_Correlation_{i}.png')
    items = []
    for x in df:
        for y in df:
            items.append((x, y, df.loc[x, y]))
    hm = hv.HeatMap(items)
    hm.opts(opts.HeatMap(colorbar=True, width=800, height=800, tools=['hover'], xrotation=90, clim=(0, 1)))
    save(hv.render(hm))

def get_representative_groups(df, clusters):
    most_repr_sets = []
    for count, cluster in enumerate(clusters):
        if len(cluster) > 2:
            output_file(filename=f"Heatmap{count}.html")
            correlation = pd.DataFrame(index=cluster, columns=cluster)
            for column in cluster:
                for column1 in cluster:
                    cor_value = dcor.distance_correlation(df[column], df[column1])
                    correlation.loc[column1, column] = cor_value
                correlation[column] = correlation[column].astype(float)
            most_repr_sets.append(correlation.sum().idxmax())
            # Make the heatmap
            plot_local_heatmap(correlation)
        else:
            most_repr_sets.append(cluster[0])
    return most_repr_sets