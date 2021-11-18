import numpy as np
import pandas as pd
import os
import dcor
import sys

def loop_over_files(directory):
    for entry in os.scandir(directory):
        if 'ICARUN' in entry.path:
            shapes = []
            for file in os.scandir(entry):
                if 'ica_run' in file.path:
                    df = pd.read_csv(file.path,
                                     sep='\t', index_col=0)
                    shapes.append(df.shape[1])
                    break
            print(f'ICA shapes of separate runs of {entry.path.split("/")[-1]} are {set(shapes)}')


def load_consensus(directory):
    for entry in os.scandir(directory):
        if 'ICARUN' in entry.path:
            for file in os.scandir(entry):
                if 'ica_independent_components_consensus.tsv' in file.path:
                    df = pd.read_csv(file, sep='\t', index_col=0)
                    print(f'ICA shapes of consensus of {entry.path.split("/")[-1]} are {df.shape[1]}')

def load_cutoff_consensus(directory, cutoff=0.5):
    for entry in os.scandir(directory):
        if 'ICARUN' in entry.path:
            for file in os.scandir(entry):
                if 'ica_robustness_metrics_independent_components_consensus.tsv' in file.path:
                    df = pd.read_csv(file, sep='\t', index_col=0)
                    df = df[df['credibility index'] > cutoff]
                    print(f'Consensus components with cutoff {cutoff} in {entry.path.split("/")[-1]} are {df.shape[0]}')
print('Non')
print('___________________________________________________________________')
directory = '/home/MarkF/DivideConquer/Results/MathExperiment/2_Split/Non_Normalized'
loop_over_files(directory)
print()
load_consensus(directory)
print()
load_cutoff_consensus(directory)
print('___________________________________________________________________')

print('1')
print('___________________________________________________________________')
directory = '/home/MarkF/DivideConquer/Results/MathExperiment/2_Split/One_Normalized'
loop_over_files(directory)
print()
load_consensus(directory)
print()
load_cutoff_consensus(directory)
print('___________________________________________________________________')

print('3')
print('___________________________________________________________________')
directory = '/home/MarkF/DivideConquer/Results/MathExperiment/2_Split/Three_Normalized'
loop_over_files(directory)
print()
load_consensus(directory)
print()
load_cutoff_consensus(directory)
print('___________________________________________________________________')

