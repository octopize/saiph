from itertools import repeat
from typing import List

import numpy as np
import pandas as pd

from saiph.models import DFLike


def column_names(n: int) -> List[str]:
    return [f"Dim. {i + 1}" for i in range(n)]


def row_weights_uniform(n: int) -> List[float]:
    return [k for k in repeat(1 / n, n)]


def explain_variance(s: DFLike, df: pd.DataFrame, nf: int):
    explained_var = ((s ** 2) / (df.shape[0] - 1))[:nf]  # type: ignore
    summed_explained_var = explained_var.sum()
    if summed_explained_var == 0:
        explained_var_ratio = np.nan
    else:
        explained_var_ratio = explained_var / explained_var.sum()
    return explained_var, explained_var_ratio
