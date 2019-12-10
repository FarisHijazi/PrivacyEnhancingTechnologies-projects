import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import dp_kmeans
import utils


df_test = np.array([
    [0, 0],
    [1, 1],
    [100, 100],
    [101, 101],
    [1000, 1000],
    [1001, 1001],
])

# for i in range(df_test.shape[0]):
#     df_test[i, :] *= i

print(df_test)

# dp kmeans
n_clusters = 2
eps = 100

print('DP Kmeans eps=', eps)
centroids_dp, labels_dp = dp_kmeans.kmeans(df_test, eps=eps, n_clusters=n_clusters, plot=True, seed=None, verbose=False)
