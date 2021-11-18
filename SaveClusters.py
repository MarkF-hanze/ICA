import pandas as pd

split = '4_Split'
df = pd.read_csv(f'/home/MarkF/DivideConquer/Results/Math_Clustered/{split}/HDDCClusters.csv', index_col=0)
all_data = pd.read_csv(
    '/home/MarkF/DivideConquer/Results/MathExperiment/2_Split/One_Normalized/Math_ExpAll.csv', index_col=0)
all_data['Clusters'] = df.iloc[:,0].values
for number, number_df in all_data.groupby('Clusters'):
    number_df = number_df.drop('Clusters', axis=1)
    number_df.to_csv(f'/home/MarkF/DivideConquer/Results/Math_Clustered/{split}/Math_Exp{number}.csv')
