import pandas as pd
import numpy as np
import holoviews as hv
import sys
from bokeh.palettes import Blues256, Category20, Category10
from holoviews import opts
from collections import defaultdict
from bokeh.io import export_png, export_svgs
hv.extension('bokeh')
pd.set_option('display.max_columns', None)

# Create a citrus plot to see the correlation between different subsets Kwargs is given to the plot function
class CitrusPlot(object):
    def __init__(self, correlation, line_width_small, line_width_big, saver, node_color_palette=None, fake_amount=1000,
                 fontscale=1, **kwargs):
        self.fontscale = fontscale
        # Make all the variables global
        self.saver = saver
        self.fake_amount = fake_amount
        self.line_width_small = line_width_small
        self.line_width_big = line_width_big
        self.kwargs = kwargs
        self.correlation = correlation
        self.lines = None
        self.nodes = None
        # Remove diagonal and values smaller than cutoff
        self.correlation.values[[np.arange(correlation.shape[0])] * 2] = np.nan
        # Count the unique groups and for the fake lines
        # Loop over every fake
        self.current_fake = 0
        # Colorpallet
        self.color_palette_high = Blues256
        self.color_palette_low = '#D0D0D0'
        self._create_color_palette_lines()
        # Create the lines
        self._create_lines()
        # Create the nodes
        self._create_nodes()
        # Draw one line everywhere so set the value to one and make it a int so holoviews interprets it correct
        self.lines['value'] = 1
        self.lines['value'] = self.lines['value'].astype(int)
        # Create white empty space nodes between the groups
        self._create_fake_nodes()
        # Create the node color pallet
        self.node_color_palette = node_color_palette
        self._create_color_palette_nodes()
        # Set the node color
        self.nodes['node_color'] = [self.node_color_palette[z] for z in self.nodes.Group]
        # Set all the nodes to have the same line width

    def _create_color_palette_lines(self):
        # Make until 0.6 grey and the rest a shade of blue
        # Get the greys
        size = len(list(self.color_palette_high))
        self.color_pallet_lines = list(np.repeat(self.color_palette_low, round((size / 4) * 6)).tolist())
        # Add the colors
        self.color_pallet_lines.extend(list(Blues256)[::-1])

    def _create_lines(self):
        # Melt the dataframe to 3 columns
        self.lines = self.correlation.reset_index().melt(id_vars='index').dropna()
        # Drop the duplicates that like ab ba because correlation was mirrored
        self.lines = self.lines.loc[
            pd.DataFrame(np.sort(self.lines[['index', 'variable']], 1), index=self.lines.index).drop_duplicates(
                keep='first').index]
        # Remove lines that go from 1 group to the same group. (Line from subset1 to subset1)
        self.lines = self.lines[
            self.lines['index'].str.split('_', expand=True)[1] != self.lines['variable'].str.split('_', expand=True)[1]]
        # Set the color and the alpha to the strength of the correlation
        self.lines['color'] = self.lines['value']
        self.lines['alpha'] = self.lines['value']
        # Math experiment
        self.lines['width'] = [self.line_width_small if z < 0.6 else self.line_width_big for z in self.lines['value']]
        # Cancer settings
        # lines['width'] = [.08 if z < 0.6 else .9 for z in lines['value']]
        # Remove the low lines to up the performance
        self.remove_lines(0.05)
        # Sort
        self.lines = self.lines.sort_values('color')

    def _create_nodes(self):
        # Create a node for every component
        self.nodes = pd.DataFrame()
        self.nodes['Components'] = self.correlation.columns
        # Add it to the group for color
        self.nodes['Group'] = [z.split('_')[1] for z in self.correlation.columns]
        # Do the sorting based on how many scores it has
        self.get_score_nodes(True)
        self.get_score_nodes(False)
        self.nodes = self.nodes.sort_values(['Group', 'value_y', 'value_x'], ascending=True)
        self.nodes = self.nodes.drop('value_x', axis=1)
        self.nodes = self.nodes.drop('value_y', axis=1)
        # Add text to every middle value
        # Loop over every group and add text to the middle node
        new_nodes = pd.DataFrame()
        for group, df in self.nodes.groupby('Group'):
            group_text = list(np.repeat('', df.shape[0]))
            if group == 'big':
                group = 'Sample'
            else:

                group = f'Subset {group}'
                #group = f'{group}'
            group_text[df.shape[0] // 2] = group
            df['text'] = group_text
            new_nodes = new_nodes.append(df)
        self.nodes = new_nodes

    def get_score_nodes(self, above_cutoff):
        score = self.lines.copy()[['index', 'variable', 'value']]
        # Count how often high and low correlations occur for every point
        if above_cutoff:
            a = np.array(score['value'].values.tolist())
            score['value'] = np.where(a >= 0.6, 0, a).tolist()
        else:
            a = np.array(score['value'].values.tolist())
            score['value'] = np.where(a < 0.6, 0, a).tolist()
        # Sum the score for every component
        total_score = defaultdict(lambda: 0)
        for index, row in score.iterrows():
            total_score[row['index']] = total_score[row['index']] + row['value']
            total_score[row['variable']] = total_score[row['variable']] + row['value']
        score = pd.DataFrame(list(total_score.items()), columns=['Components', 'value'])
        self.nodes = pd.merge(left=self.nodes, right=score, left_on='Components', right_on='Components')

    def remove_lines(self, cutoff):
        # Remove lines that have a correlation strength of smaller than cutoff
        all_nodes = set(self.lines[['index', 'variable']].values.ravel())
        self.lines = self.lines[self.lines['value'] > cutoff]
        # Check if all nodes still have at least 1 line otherwise holoviews will give a error
        nodes_with_lines = set(self.lines[['index', 'variable']].values.ravel())
        # Nodes that dont have a line
        missing_nodes = all_nodes.difference(nodes_with_lines)
        self.create_fake_lines(list(missing_nodes))

    def create_fake_lines(self, nodes):
        group_length = self.correlation.columns
        group_length = [x.split('_')[1] for x in group_length]
        group_length = len(set(group_length)) - 1
        # Add the nodes back with a fake line
        for i in range(0,len(nodes)):
            # Just find a node to connect the line to. Doenst matter it is invisible anyway
            # Make the item in the lines dataframe with zeros every were put in here the values that are not zero
            try:
                row = {
                    'index': nodes[i],
                    'variable': nodes[i+1],
                    'value': 1,
                    'width': self.line_width_small,
                }
            except IndexError:
                row = {
                    'index': nodes[i],
                    'variable': nodes[0],
                    'value': 1,
                    'width': self.line_width_small,
                }
            if self.current_fake < group_length:
                self.current_fake += 1
            else:
                self.current_fake = 0
            for column in self.lines:
                if column not in row:
                    row[column] = 0
            self.lines = self.lines.append(row, ignore_index=True)

    def _create_fake_nodes(self):
        new_nodes = pd.DataFrame()
        all_fakes = pd.DataFrame()
        i = 0
        # Loop over every group
        for group, df in self.nodes.groupby('Group'):
            # Create the fake nodes
            fake_df = pd.DataFrame()
            # Give them a name and fill everything in
            fake_df['Group'] = np.repeat(f'fake{i}', self.fake_amount)
            fake_df['Components'] = ['consensus independent component ' + str(x) +
                                     f'_fake{i}' for x in range(0, self.fake_amount)]
            fake_df['text'] = ''
            # Add the fake df between all the groups
            new_nodes = pd.concat([new_nodes, df, fake_df])
            all_fakes = pd.concat([fake_df, all_fakes])
            i += 1
        self.nodes = new_nodes
        self.create_fake_lines(all_fakes['Components'].values)

    def _create_color_palette_nodes(self):
        i = len(self.node_color_palette)
        # Give every group without a color a color
        for group in self.nodes.Group.unique():
            if group not in self.node_color_palette:
                # Fake is colored white
                if 'fake' in group:
                    self.node_color_palette[group] = '#FFFFFF'
                # TODO deze set nog globaal invulbaar maken
                # ELse from catergory 20 like the other group
                else:
                    self.node_color_palette[group] = Category20[20][i]
                    i += 1


    def plot(self):
        # Make nodes a holoview object
        nodes = hv.Dataset(self.nodes, 'Components', ['Group', 'node_color', 'text'])
        # Make chord
        chord = hv.Chord((self.lines, nodes))
        # Start plotting with the set parameters
        chord.opts(
            opts.Chord(edge_color='color', edge_cmap=self.color_pallet_lines, edge_alpha='alpha',
                       labels='text', node_color='node_color', label_text_color='node_color',
                       edge_line_width='width', node_marker='none',
                       colorbar_position='top', **self.kwargs

        ))
        chord.opts(fontscale=self.fontscale)
        chord = chord.redim.range(color=(0, 1))
        hv.save(chord, f'{self.saver.get_path()}/citrusPlot.png')
        #hv.save(chord, f'{self.saver.get_path()}/citrusNormAll-new.png')

