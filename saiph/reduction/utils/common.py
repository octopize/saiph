from itertools import repeat
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy
from numpy.typing import NDArray


def column_names(n: int) -> List[str]:
    return [f"Dim. {i + 1}" for i in range(n)]


def row_weights_uniform(n: int) -> NDArray[np.float64]:
    return np.array([k for k in repeat(1 / n, n)])


def diag(arr: NDArray[Any], use_scipy: bool = False) -> NDArray[Any]:
    if use_scipy:
        return scipy.sparse.diags(arr)  # type: ignore
    else:
        return np.diag(arr)  # type: ignore


def explain_variance(
    s: NDArray[Any], df: pd.DataFrame, nf: int
) -> Tuple[NDArray[Any], Union[NDArray[Any], float]]:
    explained_var: NDArray[Any] = ((s ** 2) / (df.shape[0] - 1))[:nf]
    summed_explained_var = explained_var.sum()
    if summed_explained_var == 0:
        explained_var_ratio = np.nan
    else:
        explained_var_ratio = explained_var / explained_var.sum()
    return explained_var, explained_var_ratio
