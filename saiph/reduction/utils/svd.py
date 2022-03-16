from typing import Any, Tuple

import pandas as pd
from numpy.typing import NDArray
from scipy import linalg
from sklearn.utils import extmath
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

def SVD(
    df: pd.DataFrame, svd_flip: bool = True, algorithm: str = 'randomized'
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Compute Singular Value Decomposition.

    Parameters
    ----------
    df: pd.DataFrame
        Matrix to decompose.
    svd_flip: bool
        Whether to use svd_flip on U and V or not.

    Returns
    -------
    U: np.ndarray
        Unitary matrix having left singular vectors as columns.
    s: np.ndarray
        The singular values.
    V: np.ndarray
        Unitary matrix having right singular vectors as rows.
    """
    if algorithm == 'lapack':
        U, s, V = linalg.svd(df, full_matrices=False)
        if svd_flip:
            U, V = extmath.svd_flip(U, V)
    
    if algorithm == 'randomized':
        X = csr_matrix(df)
        print(X.shape)
        svd = TruncatedSVD(n_components=X.shape[1] - 1, n_iter=7, random_state=42)
        U=svd.fit_transform(X)
        V=svd.components_
        s=svd.singular_values_ 

    return U, s, V
