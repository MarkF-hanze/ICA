import sys

from bokeh.palettes import Category10

from HelperClasses.Correlation import Correlation
from HelperClasses.LoadData import LoadICARuns, LoadCancer
from HelperClasses.BiologicalInterpertation import MergeTwo, Histogram, BigSmall
from HelperClasses.CitrusPlot import CitrusPlot
from HelperClasses.Heatmap import Heatmap
import holoviews as hv
import random

hv.extension('bokeh')


class Saver(object):
    def __init__(self, path):
        self.path = path

    def get_path(self):
        return self.path


if __name__ == "__main__":

    method = 'Clustered'
    splits = '4_Split'
    cancer_type = True
    # Load the small and big data
    if cancer_type:
        datasets = LoadCancer('/home/MarkF/DivideConquer/Results/GPL570/',
                              '/home/MarkF/DivideConquer/Results/GPL570/All_Cancer/ICARUN/'
                              'ica_independent_components_consensus.tsv')
        saver = Saver(f'Results/Cancer_type')
        node_pallete = {'big': '#000000'}
    else:
        node_pallete = {'big': '#000000',
                        '1': Category10[10][0],
                        '2': Category10[10][2],
                        '3': Category10[10][4],
                        '4': Category10[10][1]}
        datasets = LoadICARuns(
            f'/home/MarkF/DivideConquer/Results/2000_Samples_Experiment/Clustered_vs_Random_Experiment/'
            f'{method}_Splits/{splits}',
            '/home/MarkF/DivideConquer/Results/2000_Samples_Experiment/Clustered_vs_Random_Experiment/'
            'ICARUN_ALL/ica_independent_components_consensus.tsv')
        saver = Saver(f'/home/MarkF/DivideConquer/ICA/Results/{method}/{splits}')
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
    # Make the plotter and set the colors
    plotter = Histogram({'big': (0, 0, 0)}, saver)
    # Plot the highest scores with the maximum correlation analysis
    biologicalInt = MergeTwo(correlation.get_correlation())
    biologicalInt.plot(plotter, 0.5)
    biologicalInt = BigSmall(correlation.get_correlation())
    biologicalInt.plot(plotter)
    # Turn it to html colors for later
    # Turn it to later colors for the html plot
    color_mapper_html = {}
    for z in plotter.get_colormap():
        rgb = tuple([int(x * 255) for x in plotter.get_colormap()[z]])
        color_mapper_html[z] = '#%02x%02x%02x' % rgb[:3]

    citrusplotter = CitrusPlot(correlation.get_correlation(),
                               node_color_palette=node_pallete,
                               line_width_small=3, line_width_big=9, fake_amount=1000,
                               height=7000, width=7000, node_radius=1, label_text_font_size='200px',
                               colorbar_opts={'width': 5000, 'height': 150,
                                              'title': "Pearson's correlation",
                                              'label_standoff': 20,
                                              }, fontscale=20, colorbar=False,
                               saver=saver)
    # itrusplotter = CitrusPlot(correlation.get_correlation(), node_color_palette=color_mapper_html,
    #                           line_width_small=.3, line_width_big=.9, fake_amount=1000,
    #                           height=700, width=700, node_radius=1, label_text_font_size='40px',
    #                           colorbar_opts={'width': 500, 'title': 'Pearson correlation'}, saver=saver)
    citrusplotter.plot()
    sys.exit()
    # Cluster and plot
    heatmap = Heatmap(correlation.get_correlation(), cut_off, saver)
    heatmap.plot()
    # Make the clusters based on the correlation
    heatmap.make_clusters()
    # heatmap.set_cluster(fake_clusters)
    # Give it the normal estimated sources
    heatmap.merge_clusters(correlation.get_merged_normall())
