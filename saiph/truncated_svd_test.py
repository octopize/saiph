from dataclasses import replace
from saiph.reduction.utils.svd import SVD

from saiph.reduction.utils.common import explain_variance
import numpy as np

import time
import pandas as pd

N_ROWS = 10000
N_COLS = 1000


def create_big_iris( ) -> pd.DataFrame :
        
    iris = pd.read_csv('../fixtures/iris.csv')

    continuous_iris = iris.drop(columns=["variety"])
    big_iris = np.random.default_rng(42).choice(continuous_iris, size=N_ROWS, replace=True)
    big_iris = pd.DataFrame(big_iris, columns=continuous_iris.columns)
    original_big_iris = big_iris.copy()

    ncols = original_big_iris.shape[1]
    while ncols < N_COLS:

        iris_copy = big_iris.copy()
        big_iris = pd.concat([big_iris, iris_copy], axis="columns")
        ncols = big_iris.shape[1]

    print(f"Big iris now of shape {big_iris.shape}")

    return big_iris


df = create_big_iris()
df_centered = (df - df.mean(axis=0)/df.std(axis=0))

# n_components = [10, 100, 999]
# approximate = True
# algorithm = 'arpack'

# for comp in n_components:
#     start = time.time()
#     SVD(df_centered, approximate=approximate, n_components=comp, algorithm=algorithm)
#     print("Time : ", time.time() - start)

approximate = False
start = time.time()
U,s,V = SVD(df_centered, approximate=approximate, n_components=None, algorithm=None)

print(U.shape)
explained_var, explained_var_ratio = explain_variance(s, df, df.shape[1])
print("Explained variance :", explained_var[:10])
print("Explained variance  ratio:", explained_var_ratio[:10])
print("cumsum Explained variance ratio:", np.cumsum(explained_var_ratio[:10]))
print("Time : ", time.time() - start)
