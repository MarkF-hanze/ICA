import numpy as np

from consensus import load_data
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

directorys = [
    ['/home/MarkF/DivideConquer/Results/MathExperiment/2_Split/One_Normalized',
     '/home/MarkF/DivideConquer/Results/MathExperiment/3_Split/',
     '/home/MarkF/DivideConquer/Results/MathExperiment/4_Split/'],
    ['/home/MarkF/DivideConquer/Results/Math_Clustered/2_Split/',
     '/home/MarkF/DivideConquer/Results/Math_Clustered/3_Split/',
     '/home/MarkF/DivideConquer/Results/Math_Clustered/4_Split/']
]

xs = [2, 3, 4]
fig, ax = plt.subplots(2, 1, sharex=True)
i = 0
labels = {
    'Random split': 'cyan',
    'Clustered split': 'red'
}
for label, split_type in zip(labels, directorys):
    print(label)
    ys_small = []
    ys_big = []
    for directory in split_type:
        print(directory)
        small_data, big_data, split_columns = load_data(directory,
                                                        '/home/MarkF/DivideConquer/Results/MathExperiment'
                                                        '/0_Credibility/ica_independent_components_consensus.tsv')
        # Merge the small components
        small_non_appearing_columns = []
        small_appearing_columns = []
        big_appearing_columns = []
        for df_small in small_data:
            #df_small = df_small.sample(frac=0.1, axis=1)
            for column in tqdm(df_small.columns):
                consensus = False
                for column1 in big_data:
                    if abs(pearsonr(df_small[column], big_data[column1])[0]) > 0.8:
                        consensus = True
                        big_appearing_columns.append(column1)
                        break
                if consensus:
                    small_non_appearing_columns.append(column)
                else:
                    small_appearing_columns.append(column)
        big_non_appearing_columns = [x for x in big_data.columns if x not in big_appearing_columns]
        ys_small.append(len(small_non_appearing_columns))
        ys_big.append(len(big_non_appearing_columns))
    ax.ravel()[0].plot(xs, ys_small, label=label, color=labels[label])
    ax.ravel()[0].scatter(xs, ys_small, color=labels[label])
    ax.ravel()[1].plot(xs, ys_big, label=label, color=labels[label])
    ax.ravel()[1].scatter(xs, ys_big, color=labels[label])
    i += 1

for plot in ax.ravel():
    plot.set_ylabel('Count')
ax.ravel()[1].set_xlabel('Amount of splits')
ax.ravel()[0].set_title("Small non appearing estimated sources in big")
ax.ravel()[1].set_title("Big non appearing estimated sources in small")
ax.ravel()[1].legend(shadow=True, fancybox=True)
ax.ravel()[1].legend(bbox_to_anchor=(-0.01, 1.2))
# ax.ravel()[2].set_title("Small non appearing clusters in big clustered split")
# ax.ravel()[3].set_title("Big non appearing clusters in small clustered split")
plt.show()
