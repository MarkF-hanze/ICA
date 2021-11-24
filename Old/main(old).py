import pandas as pd
import scipy.stats as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
pd.options.mode.chained_assignment = None  # default='warn'


class Pipeline:
    def __init__(self):
        # Read the GSEA outcome
        self.metabolic = None
        self.immune_df = None
        # Load the files
        df_reactome = pd.read_csv('/home/MarkF/DivideConquer/Results/GPL570/Adrenal_cancer/GSEA'
                                  '/enrichment_matrix_Reactome.tsv', sep='\t')
        df_gobp = pd.read_csv('/home/MarkF/DivideConquer/Results/GPL570/Adrenal_cancer/GSEA/enrichment_matrix_Gene '
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

    @staticmethod
    def analyse_results(df):
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
                        'is_DNA_Repair': 0
                    }
                for imp_row in imp_rows:
                    counts[comp][imp_row] += row[imp_row]
        return counts

    def compare(self):
        pd.set_option('display.max_columns', None)
        entrez_id = pd.read_csv('/home/MarkF/DivideConquer/Tacna_confic/'
                                'Genomic_Mapping_hgu133plus2_using_jetscore_30032018.txt', sep='\t', index_col=1)
        ces = pd.read_csv('/home/MarkF/DivideConquer/Results/GPL570/Leukemia/ICARUN/'
                         'ica_independent_components_consensus.tsv', sep='\t', index_col=0)

        ces1 = pd.read_csv('/home/MarkF/DivideConquer/Results/GPL570/Adrenal_cancer/ICARUN/'
                          'ica_independent_components_consensus.tsv', sep='\t', index_col=0)

        hallmark_df = pd.read_csv('/home/MarkF/DivideConquer/Results/GPL570/Adrenal_cancer/GSEA'
                                  '/enrichment_matrix_Hallmark.tsv', sep='\t')
        hallmark_df['Unnamed: 0'] = hallmark_df['Unnamed: 0'].str.split(' -- ', expand=True)[0]
        hallmark_df = hallmark_df.set_index('Unnamed: 0')
        hallmark_df = hallmark_df['consensus independent component 5']
        #print(hallmark_df.sort_values(ascending=False))


        # TODO does sign matter?
        ces = ces.abs()
        ces = pd.merge(left=ces, right=entrez_id, how='left', left_index=True, right_index=True)
        ces = ces.sort_values(['CHR_Mapping', 'BP_Mapping'])
        count = self.analyse_results(self.df_reactome)
        needed_components_dna = []
        for component in count:
            if count[component]['is_DNA_Repair'] != 0:
                print(component)
                print(count[component])
                needed_components_dna.append(component)
        needed_components_dna = ['consensus independent component 2']
        #dna_ces = ces[needed_components_dna]
        f, axs = plt.subplots(len(needed_components_dna), 1, figsize=(10, 50))
        if len(needed_components_dna) == 1:
            axs = [axs]
        print('_______________________')
        x_axis = np.arange(0, ces.shape[0])
        all_genes = []
        for ax, column in zip(axs, needed_components_dna):
            line_v = ces[column].quantile(0.99)
            ax.bar(x=x_axis, height=ces[column])
            ax.hlines(line_v, 0, len(x_axis), colors='red', linestyles='dashed')
            ax.set_title(column)
            # Get the genes
            genes = ces[ces[column] >= line_v].index
            genes = [str(x) for x in genes]
            all_genes.append(set(genes))
            print(' '.join(ces[ces[column] >= line_v]['SYMBOL'].astype(str).unique()))

        intersect = (set.intersection(*all_genes))
        print(len(intersect))
        plt.subplots_adjust(hspace=0.2 + (len(needed_components_dna) - 2) / 10)
        plt.show()


class PeakAnalysis:
    def __init__(self):
        df = pd.read_csv('/home/MarkF/DivideConquer/Results/GPL570/Adrenal_cancer/CNA_TC'
                         '/_extreme_valued_regions_all_chromosomes.txt', sep='\t')
        df = df[df['extreme_value_region_status'] != 0]
        df = df[df['mappings_in_region'] >= 10]
        df['name'] = df['name'].str.split(' ', expand=True)[3].astype(int)
        print(df.head())

class Correlation:
    def __init__(self):
        df = pd.read_csv('/home/MarkF/DivideConquer/Results/GPL570/Brain_cancer/ICARUN/'
                         'ica_independent_components_consensus.tsv', sep='\t', index_col=0)
        df1 = pd.read_csv('/home/MarkF/DivideConquer/Results/GPL570/Breast_Cancer/ICARUN/'
                         'ica_independent_components_consensus.tsv', sep='\t', index_col=0)
        merged_df = pd.merge(left=df, right=df1, how='outer', left_index=True, right_index=True)
        # Compute the correlation matrix
        corr = merged_df.corr(method='pearson')
        keep_columns = [x for x in corr.columns if '_x' in x]
        keep_index = [x for x in corr.columns if '_x' not in x]
        corr = corr.loc[keep_index, keep_columns]
        corr.columns = [x.split(' ')[-1].replace('_x', '') for x in corr.columns]
        corr.index = [x.split(' ')[-1].replace('_y', '') for x in corr.index]
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(20, 50))


        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, vmax=1, vmin=-1, square=True, ax=ax)
        plt.show()



if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.compare()
    #pipeline.start_analysis()
    #peak = PeakAnalysis()
    #Correlation()
