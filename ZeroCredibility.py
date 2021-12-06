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
        small_data, big_data, split_columns = load_data(directory,
                                                        '/home/MarkF/DivideConquer/Results/MathExperiment'
                                                        '/0_Credibility/ica_independent_components_consensus.tsv')
        # Merge the small components
        small_non_appearing_columns = []
        small_appearing_columns = []
        big_appearing_columns = []
        big_non_appearing_columns = []
        for df_small in small_data:
            combined_df = pd.merge(left=df_small, right=big_data, left_index=True, right_index=True)
            correlation = np.corrcoef(combined_df.values, rowvar=False)
            correlation = np.absolute(correlation)
            correlation = pd.DataFrame(correlation, columns=combined_df.columns, index=combined_df.columns)
            correlation = correlation.loc[df_small.columns, big_data.columns]
            correlation_big = correlation.max()
            correlation_small = correlation.max(axis=1)
            big_appearing_columns.extend(list(correlation_big[correlation_big > 0.8].index))
            small_appearing_columns.extend(list(correlation_small[correlation_small > 0.8].index))
            small_non_appearing_columns.extend(list(correlation_small[correlation_small <= 0.8].index))

        big_non_appearing_columns = [z for z in big_data.columns if z not in big_appearing_columns]
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
ax.ravel()[1].legend(bbox_to_anchor=(-0.01, 1.3))
# ax.ravel()[2].set_title("Small non appearing clusters in big clustered split")
# ax.ravel()[3].set_title("Big non appearing clusters in small clustered split")
plt.xticks([2, 3, 4])
plt.tight_layout()
#plt.show()
plt.savefig("Results/Consensus_Big_vs_Small.svg")
