from typing import Union

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.utils import extmath

DFLike = Union[pd.DataFrame, np.array]


def SVD(df: DFLike):
    """Singular Value Decomposition.

    Arguments:
        X {array-like} -- matrix to decompose

    Returns:
        array -- unitary matrix having left singular vectors as columns
        array -- the singular values
        array -- unitary matrix having right singular vectors as rows
    """
    U, s, V = linalg.svd(df, full_matrices=False)
    U, V = extmath.svd_flip(U, V)
    return U, s, V
