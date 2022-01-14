import numpy as np
import pandas as pd
import os
import dcor
import seaborn as sns
import sys

from bokeh.palettes import Category20
from matplotlib import pyplot as plt
from PIL import ImageColor
from Main import Saver

from HelperClasses.CitrusPlot import CitrusPlot


class NormCheck(object):
    def __init__(self, path, name):
        self.path = path
        self.PCA_number = None
        self.global_name = name

        self.PCA_number = {}
        self.PCA()

        self.consensus = {}
        self.load_consensus()
        self.merged_consensus = None
        self.merge_consensus()

        self.credibility = {}
        self.load_credibility()

        # print(self.consensus)
        # print(self.credibility)
        # print(self.PCA_number)

    # Get all components of a single run (This is the PCA count)
    def PCA(self):
        for entry in os.scandir(self.path):
            if 'ICARUN' in entry.path:
                shapes = []
                for file in os.scandir(entry):
                    if 'ica_run' in file.path:
                        df = pd.read_csv(file.path,
                                         sep='\t', index_col=0)
                        shapes.append(df.shape[1])
                        self.PCA_number[self.get_name(entry.path)] = set(shapes).pop()
                        break

    @staticmethod
    def get_name(string):
        return string.split('/')[-1].replace('ICARUN_', '').replace('_', '')

    # Load the consensus of each split
    def load_consensus(self):
        for entry in os.scandir(self.path):
            if 'ICARUN' in entry.path:
                df = pd.read_csv(f'{entry.path}/ica_independent_components_consensus.tsv', sep='\t', index_col=0)
                self.consensus[self.get_name(entry.path)] = df

    def load_credibility(self):
        for entry in os.scandir(self.path):
            if 'ICARUN' in entry.path:
                for file in os.scandir(entry):
                    if 'ica_robustness_metrics_independent_components_consensus.tsv' in file.path:
                        df = pd.read_csv(f'{entry.path}/ica_robustness_metrics_independent_components_consensus.tsv',
                                         sep='\t', index_col=0)
                        self.credibility[self.get_name(entry.path)] = df

    def merge_consensus(self):
        first_run = True
        for name in self.consensus:
            df = self.consensus[name].copy()
            df.columns = [f'{x}_{self.global_name}.{name}' for x in df.columns]
            if first_run:
                self.merged_consensus = df
                first_run = False
            else:
                self.merged_consensus = self.merged_consensus.join(df)

    def get_consensus(self):
        return self.merged_consensus

    def get_counts(self):
        return self.PCA_number, self.consensus, self.credibility


one = NormCheck('/home/MarkF/DivideConquer/Results/2000_Samples_Experiment/Normalized_Expiriment/One_Normalized',
                'One normalization')
three = NormCheck('/home/MarkF/DivideConquer/Results/2000_Samples_Experiment/Normalized_Expiriment/Three_Normalized',
                  'Three normalizations')
none = NormCheck('/home/MarkF/DivideConquer/Results/2000_Samples_Experiment/Normalized_Expiriment/Non_Normalized',
                 'No normalization')
### Making the citrusplots
df = one.get_consensus()
df = df.join(three.get_consensus())
df = df.join(none.get_consensus())
correlation = np.corrcoef(df.values, rowvar=False)
correlation = np.absolute(correlation)
correlation = pd.DataFrame(correlation, columns=df.columns, index=df.columns)
correlation = correlation.loc[[x for x in correlation.columns if '.ALL' in x],
                              [x for x in correlation.columns if '.ALL' in x]]
correlation = correlation.loc[[x for x in correlation.columns if 'Three' not in x],
                              [x for x in correlation.columns if 'Three' not in x]]
correlation.columns = [x.replace('.ALL', '') for x in correlation.columns]
correlation.index = [x.replace('.ALL', '') for x in correlation.index]

colors = {
    'One normalization': Category20[20][0],
    'Three normalizations': Category20[20][1],
    'No normalization': Category20[20][2],
    'One.ALL': '#641e16',
    'One.SPLIT1': '#c0392b',
    'One.SPLIT2': '#e6b0aa',
    'Three.ALL': '#1b4f72',
    'Three.SPLIT1': '#3498db',
    'Three.SPLIT2': '#aed6f1',
    'None.ALL': '#186a3b',
    'None.SPLIT1': '#2ecc71',
    'None.SPLIT2': '#82e0aa',
}
saver = Saver('Results/Normalization_Experiment')
citrusplotter = CitrusPlot(correlation, node_color_palette=colors, saver=saver,
                           node_line_color='black',
                           line_width_small=.3, line_width_big=.9, fake_amount=1000,
                           height=700, width=700, node_radius=1, label_text_font_size='10px',
                           colorbar_opts={'width': 500, 'title': 'Pearson correlation'})
# citrusplotter.plot()


### Make the countplot
splits = {
    'One normalization': one.get_counts(),
    'Three normalization': three.get_counts(),
    'No normalization': none.get_counts()
}
plot_df = []
for split in splits:
    # Loop over the counts
    for name in splits[split][0]:
        plot_df.append([f'{split} {name}', 'PCA', splits[split][0][name]])
        plot_df.append([f'{split} {name}', 'ICA', splits[split][1][name].shape[1]])
        plot_df.append([f'{split} {name}', 'ICA credibility > 0.5',
                        splits[split][2][name][splits[split][2][name]['credibility index'] > 0.5].shape[0]])
plot_df = pd.DataFrame.from_records(plot_df, columns=['Bar type', 'Component type', 'Count'])
# Remove Three ALl
plot_df = plot_df[plot_df['Bar type'] != 'Three normalization ALL']
plot_df['Bar type'] = plot_df['Bar type'].str.replace('SPLIT', 'subset ')
plot_df['Bar type'] = plot_df['Bar type'].str.replace('ALL', 'sample')
plot_df['Bar type'] = pd.Categorical(plot_df['Bar type'],
                                     ['One normalization subset 1', 'Three normalization subset 1',
                                      'No normalization subset 1',
                                      'One normalization subset 2', 'Three normalization subset 2',
                                      'No normalization subset 2',
                                      'One normalization sample',  'No normalization sample'])
plot_df = plot_df.sort_values('Bar type')

#  '#2ecc71'
colors = ['#641e16', '#c0392b', '#e6b0aa',
          '#1b4f72', '#3498db', '#aed6f1',
          '#186a3b', '#82e0aa']
colors = [tuple(c / 255 for c in ImageColor.getcolor(i, "RGB")) for i in colors]
sns.barplot(data=plot_df, x='Component type', y='Count', hue='Bar type', palette=colors)
plt.savefig('Results/Normalization_Experiment/Counts.svg', dpi=300)
