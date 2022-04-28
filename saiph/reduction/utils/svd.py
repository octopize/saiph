from typing import Any, Tuple

import pandas as pd
from numpy.typing import NDArray
from scipy import linalg
from sklearn.utils import extmath


def SVD(
    df: pd.DataFrame, svd_flip: bool = True
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Compute Singular Value Decomposition.

    Parameters:
        df: Matrix to decompose.
        svd_flip: Whether to use svd_flip on U and V or not.

    Returns:
        U: Unitary matrix having left singular vectors as columns.
        s: The singular values.
        V: Unitary matrix having right singular vectors as rows.
    """
    U, s, V = linalg.svd(df, full_matrices=False)
    if svd_flip:
        U, V = extmath.svd_flip(U, V)
    return U, s, V
