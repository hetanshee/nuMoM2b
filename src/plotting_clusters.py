import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs

csv = Path('Research Assistant').with_name('Markers.CSV')  # exported Hetanshees cluster dataset
df = pd.read_csv(csv)

def plottin(dataf):
        copy = pd.DataFrame()
        copy = dataf.groupby('clusters')[
                ['SubEDINScore', 'SubSTAIScore', 'Suicidality', 'CMAE04a1a', 'CMAE04a1b', 'CMAE04a1c', 'CMAE04a2a',
                 'CMAE04a2b', 'CMAE04a2c', 'ADHD', 'OCD', 'panic disorder']].mean()
        copy = copy.T
        print(copy)
        copy = copy.reindex(
                ['SubEDINScore', 'SubSTAIScore', 'CMAE04a1a', 'CMAE04a1b', 'CMAE04a1c', 'CMAE04a2a', 'CMAE04a2b',
                 'CMAE04a2c', 'Suicidality', 'ADHD', 'OCD', 'panic disorder'])
        # re-ordering putting suicidality near the end since enorsement is low

        ax = copy.plot(figsize=(16, 5), fontsize=9, marker='o',
                       linestyle='-', title="Clusters and % Endorsement of Variables", grid=True,
                       xlabel='Variables', ylabel='Percent Endorsement')
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        ax.set_xticklabels(
                ['SubEDINScore', 'SubSTAIScore', 'CMAE04a1a', 'CMAE04a1b', 'CMAE04a1c', 'CMAE04a2a', 'CMAE04a2b',
                 'CMAE04a2c', 'Suicidality', 'ADHD', 'OCD', 'panic disorder'])
        plt.show()

plottin(df)
