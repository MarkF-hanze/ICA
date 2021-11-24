import pandas as pd
import scipy.stats as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
import dcor

pd.options.mode.chained_assignment = None
from collections import defaultdict
from bokeh.plotting import figure, show, ColumnDataSource, output_file
from bokeh.palettes import Inferno256
import holoviews as hv
from holoviews import opts, dim
from tqdm import tqdm
from consensus import create_fakes
from sklearn.preprocessing import MinMaxScaler

hv.extension('bokeh')

# TODO Z-scpre checken voor negativiteit
class Pipeline:
    def __init__(self):
        # Read the GSEA outcome
        counts = None
        self.metabolic = None
        self.immune_df = None
        # Load the files
        df_reactome = pd.read_csv('/home/MarkF/DivideConquer/Results/GPL570/Lymphoma/GSEA'
                                  '/enrichment_matrix_Reactome.tsv', sep='\t')
        df_gobp = pd.read_csv('/home/MarkF/DivideConquer/Results/GPL570/Lymphoma/GSEA/enrichment_matrix_Gene '
                              'Ontology - Biolocal Processes.tsv', sep='\t')
        df_reactome = self.clean_enrichment_matrix(df_reactome)
        df_gobp = self.clean_enrichment_matrix(df_gobp)
        # Load the different processes
        self.load_immune()
        self.load_metabolic()
        self.dna_df = pd.read_excel('/home/MarkF/DivideConquer/Tacna_confic/GOBP_REACTOME_RELEVANT_GENESETS.xlsx',
                                    header=None)
        # Add them together
        self.df_reactome = self.add_to_file(df_reactome)
        self.df_gobp = self.add_to_file(df_gobp)

    @staticmethod
    def clean_enrichment_matrix(df):
        df[['gene_set_name', 'Link']] = df['Unnamed: 0'].str.split(' --', expand=True)
        df = df.drop('Unnamed: 0', axis=1)
        return df

    def load_immune(self):
        pd.read_csv('/home/MarkF/DivideConquer/Tacna_confic/reactome_immune_gene_sets.tsv')
        immune_df_reactome = pd.read_csv('/home/MarkF/DivideConquer/Tacna_confic/reactome_immune_gene_sets.tsv',
                                         sep='\t')
        immune_df_gobp = pd.read_csv(
            '/home/MarkF/DivideConquer/Tacna_confic/gobp_immune_gene_sets_manualselection_updated.tsv', sep='\t')
        immune_df = pd.concat([immune_df_reactome, immune_df_gobp])
        self.immune_df = immune_df[immune_df['is_immune_process'] == True]

    def load_metabolic(self):
        metabolic = pd.read_csv('/home/MarkF/DivideConquer/Tacna_confic/metabolic_gene_sets.txt', sep='\t')[
            'GO_NUCLEOBASE_METABOLIC_PROCESS']
        self.metabolic = metabolic.str.replace('GO', 'GOBP')

    def add_to_file(self, df):
        df = pd.merge(df, self.immune_df, left_on='gene_set_name', right_on='gene_set_name', how='left')
        df['is_metabolic_process'] = df['gene_set_name'].isin(self.metabolic)
        df['is_DNA_Repair'] = df['gene_set_name'].isin(self.dna_df[0])
        # Only keep that has at least one True
        df['keep'] = ((df['is_immune_process'] == True) | (df['is_metabolic_process'] == True) | (
                df['is_DNA_Repair'] == True))
        return df

    def start_analysis(self):
        count = self.analyse_results(self.df_reactome)
        print(f'Reactome {count}')
        count = self.analyse_results(self.df_gobp)
        print(f'GOBP {count}')

    def analyse_results(self, df):
        # Drop the not relevant columns
        test_df = df.drop(
            ['gene_set_name', 'Link', 'is_immune_process', 'is_metabolic_process', 'is_DNA_Repair', 'keep'],
            axis=1)
        shape = test_df.shape
        # TODO What is the number of tests?
        # Bonferroni correction
        value = st.norm.ppf(0.05 / (shape[0] * shape[1]))
        test_df = test_df.abs()
        test_df = (test_df > abs(value))
        # See of any value in the row is still significant (So higher than corrected value) and only keep those
        df['Any_Sig'] = test_df.sum(axis=1) > 0
        test_df = test_df[test_df.sum(axis=1) > 0]
        # Loop over the significant rows and get those column names (The names of the components)
        significant_components = {}
        for index, row in test_df.iterrows():
            significant_components[index] = list(row[row == True].index)
        # Only keep the rows that have both a significant value and this significant value is in a relevant gene for the
        # gene sets
        biology_df = df[((df['Any_Sig'] == True) & (df['keep'] == True))]
        # Add the signifcant components and change Nan to False
        biology_df.loc[:, "Significant"] = pd.Series(significant_components)
        biology_df = biology_df.fillna(False)
        # Count how often each independent components has a gene that is linked to a gene set that is linked to a
        # process
        counts = {}
        imp_rows = ['is_immune_process', 'is_metabolic_process', 'is_DNA_Repair']
        for index, row in biology_df.iterrows():
            for comp in row['Significant']:
                if comp not in counts:
                    counts[comp] = {
                        'is_immune_process': 0,
                        'is_metabolic_process': 0,
                        'is_DNA_Repair': 0,
                        'is_CNA': 0
                    }
                for imp_row in imp_rows:
                    counts[comp][imp_row] += row[imp_row]
        cna_components = self.peak_analysis()
        for comp in cna_components:
            if comp in counts:
                counts[comp]['is_CNA'] = cna_components[comp]
            else:
                counts[comp] = {
                    'is_immune_process': 0,
                    'is_metabolic_process': 0,
                    'is_DNA_Repair': 0,
                    'is_CNA': cna_components[comp]
                }
        return counts

    def peak_analysis(self):
        df = pd.read_csv('/home/MarkF/DivideConquer/Results/GPL570/Lymphoma/CNA_TC'
                         '/_extreme_valued_regions_all_chromosomes.txt', sep='\t')
        df = df[df['extreme_value_region_status'] != 0]
        df = df[df['mappings_in_region'] >= 10]
        # df['name'] = df['name'].str.split(' ', expand=True)[3].astype(int)
        consensus_values = defaultdict(lambda: 0)
        for index, row in df.iterrows():
            consensus_values[row['name']] = consensus_values[row['name']] + 1
        return consensus_values

    def mixing_matrix(self):
        df = pd.read_csv('/home/MarkF/DivideConquer/Results/GPL570/Lymphoma/ICARUN/'
                         'ica_mixing_matrix_consensus.tsv', sep='\t', index_col=0)
        self.counts = self.analyse_results(self.df_reactome)
        print(self.counts)
        sys.exit()
        #self.citrus_plot(df)
        df = df.T
        df['Group'] = [''.join([x for x in self.counts[comp] if self.counts[comp][x] != 0])
                       if comp in self.counts else '' for comp in df.index]
        df = df[df['Group'].str.contains('is_immune_process')]
        df = df.drop('Group', axis=1)
        #print(df.corr(method=dcor.distance_correlation))
        #g = sns.clustermap(df, cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True), center=0)
        mixing = df.values
        ec = pd.read_csv('/home/MarkF/DivideConquer/Results/GPL570/Lymphoma/ICARUN/'
                         'ica_independent_components_consensus.tsv', sep='\t', index_col=0)
        ec = ec[df.index].values
        print(ec.shape)
        print(mixing.shape)
        print((ec @ mixing).shape)
        #plt.show()
        sys.exit()


    def citrus_plot(self, df):
        output_file(filename=f"citrusPlot.html")
        correlation = pd.DataFrame(index=df.columns, columns=df.columns)
        for column in tqdm(correlation):
            for column1 in correlation:
                correlation.loc[column, column1] = dcor.distance_correlation(df[column], df[column1])
        # Remove diagonal and
        correlation.values[[np.arange(correlation.shape[0])] * 2] = 0
        # Count the important correlations
        sumdf = correlation.copy()
        sumdf[sumdf < 0.5] = 0
        sumdf = sumdf.sum()
        # Count the non important columns
        sumdf1 = correlation.copy()
        sumdf1[sumdf1 > 0.5] = 0
        sumdf1 = sumdf1.sum()
        # Melt the dataframe to 3 columns
        lines = correlation.reset_index().melt(id_vars='index').dropna()
        # Drop the duplicates that like ab ba because correlation was mirrored
        lines = lines.loc[
            pd.DataFrame(np.sort(lines[['index', 'variable']], 1), index=lines.index).drop_duplicates(
                keep='first').index]
        # Create a node for every component
        lines['value'] = lines['value'].astype(float)
        #lines['color'] = [0 if x <= 0.5 else x for x in lines['value']]
        lines['color'] = [x for x in lines['value']]
        lines['alpha'] = [x for x in lines['value']]
        lines['width'] = [.5 if x <= 0.5 else .5 for x in lines['value']]
        # Scale alpha between 0.1 and 1
        lines['alpha'] = MinMaxScaler(feature_range=(0, 1)).fit_transform(lines['alpha'].values.reshape(-1, 1))
        lines['value'] = 1
        lines['value'] = lines['value'].astype(int)
        color_pallet = np.array(Inferno256)
        color_pallet[0:len(color_pallet)//2] = '#D0D0D0'
        color_pallet = list(color_pallet)
        nodes = pd.DataFrame()
        nodes['Components'] = correlation.columns
        groups = []
        for index, row in nodes.iterrows():
            comp = row['Components']
            if comp in self.counts:
                groups.append(''.join([x for x in self.counts[comp] if self.counts[comp][x] != 0]))
            else:
                groups.append('None')
        nodes['Group'] = groups
        nodes['Group'] = nodes['Group'].str.replace('is_', '')
        nodes['Group'] = nodes['Group'].str.replace('DNA_Repair', 'DNA_')
        nodes['Group'] = nodes['Group'].str.replace('immune_process', 'immune_')
        nodes['Group'] = nodes['Group'].str.replace('metabolic_process', 'metabolic_')
        nodes['Group'] = [x[:-1] if x[-1] == '_' else x for x in nodes['Group']]
        sumdf.name = 'value'
        nodes = pd.merge(left=nodes, right=sumdf, left_on='Components', right_index=True)
        sumdf1.name = 'value1'
        nodes = pd.merge(left=nodes, right=sumdf1, left_on='Components', right_index=True)
        nodes = nodes.sort_values(['Group', 'value', 'value1'], ascending=True)
        nodes = nodes.drop('value', axis=1)
        nodes = nodes.drop('value1', axis=1)
        nodes, lines = create_fakes(nodes, lines, 1000)
        lines = lines.sort_values('color')
        # Make it holoview objects
        nodes = hv.Dataset(nodes, 'Components', ['Group', 'node_color', 'text'])
        chord = hv.Chord((lines, nodes))
        chord.opts(
            opts.Chord(colorbar=True, cmap='Dark2', edge_color='color', edge_cmap=color_pallet, edge_alpha='alpha',
                       height=700, width=700, label_text_font_size='10px',
                       edge_line_width='width', node_marker='none', node_radius=5,
                       labels='text', node_color='node_color', label_text_color='node_color',
                       colorbar_position='top', colorbar_opts={'width': 500,
                                                               'title': 'Distance correlation'}))

        chord = chord.redim.range(color=(0, 1))
        p = hv.render(chord)
        p.min_border_left, p.min_border_right = 15, 15
        show(hv.render(chord))


if __name__ == "__main__":
    pd.set_option('display.max_columns', 500)
    # pd.set_option('display.width', 1000)
    pipeline = Pipeline()
    #pipeline.start_analysis()
    pipeline.mixing_matrix()
