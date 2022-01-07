from HelperClasses.Correlation import Correlation
from HelperClasses.LoadData import LoadICARuns
from HelperClasses.BiologicalInterpertation import MergeTwo, Histogram, BigSmall
from HelperClasses.CitrusPlot import CitrusPlot
from HelperClasses.Heatmap import Heatmap
import holoviews as hv
import random
hv.extension('bokeh')


if __name__ == "__main__":
    # Load the small and big data
    datasets = LoadICARuns('/home/MarkF/DivideConquer/Results/2000_Samples_Experiment/Clustered_vs_Random_Experiment/'
                           'Clustered_Splits/3_Split',
                            '/home/MarkF/DivideConquer/Results/2000_Samples_Experiment/Clustered_vs_Random_Experiment/'
                            'ICARUN_ALL/ica_independent_components_consensus.tsv')
    save_directory = '/home/MarkF/DivideConquer/ICA/Results/Clustered/2_Split'
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
    plotter = Histogram({'big': (0,0,0)})
    # Plot the highest scores with the maximum correlation analysis
    biologicalInt = MergeTwo(correlation.get_correlation())
    biologicalInt.plot(plotter, 0.5)
    biologicalInt = BigSmall(correlation.get_correlation())
    biologicalInt.plot(plotter)
    # Turn it to html colors for later
    # TODO in function
    # Turn it to later colors for the html plot
    color_mapper_html = {}
    for z in plotter.get_colormap():
        rgb = tuple([int(x * 255) for x in plotter.get_colormap()[z]])
        color_mapper_html[z] = '#%02x%02x%02x' % rgb[:3]
    # TODO size is not representative of each node because it is sized based on the line width
    citrusplotter = CitrusPlot(correlation.get_correlation(), node_color_palette=color_mapper_html,
                               line_width_small=.3, line_width_big=.9, fake_amount=1000,
                               height=700, width=700, node_radius=1, label_text_font_size='40px',
                               colorbar_opts={'width': 500, 'title': 'Pearson correlation'})
    citrusplotter.plot()

    # Cluster and plot
    heatmap = Heatmap(correlation.get_correlation(), cut_off)
    heatmap.plot()
    # Make the clusters based on the correlation
    heatmap.make_clusters()
    # Give it the normall estimated sources
    heatmap.merge_clusters(correlation.get_merged_normall())


