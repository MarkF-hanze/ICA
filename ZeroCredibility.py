from consensus import load_data
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

directorys = ['/home/MarkF/DivideConquer/Results/MathExperiment/2_Split/One_Normalized',
              '/home/MarkF/DivideConquer/Results/MathExperiment/3_Split/',
              '/home/MarkF/DivideConquer/Results/MathExperiment/4_Split/']

# directorys = ['/home/MarkF/DivideConquer/Results/Math_Clustered/2_Split/',
#               '/home/MarkF/DivideConquer/Results/Math_Clustered/3_Split/',
#               '/home/MarkF/DivideConquer/Results/Math_Clustered/4_Split/']
ys = [2, 3, 4]
xs = []
for directory in directorys:
    small_data, big_data = load_data(directory, '/home/MarkF/DivideConquer/Results/MathExperiment/0_Credibility/'
                                                'ica_independent_components_consensus.tsv')
    # Merge the small components
    non_appearing_columns = []
    appearing_columns = []
    for df_small in small_data:
        for column in df_small.columns:
            consensus = False
            for column1 in big_data:
                if abs(pearsonr(df_small[column], big_data[column1])[0]) > 0.8:
                    consensus = True
                    break
            if consensus:
                appearing_columns.append(column)
            else:
                non_appearing_columns.append(column)
    xs.append(len(non_appearing_columns))
    print(non_appearing_columns)
plt.plot(ys, xs)
plt.scatter(ys, xs)
plt.show()