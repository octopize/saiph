from typing import Any, Tuple, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import linalg
from sklearn.utils import extmath
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

def SVD(
    df: pd.DataFrame,*, approximate : bool = False, n_components : Optional[int] = None, algorithm: str = 'arpack', svd_flip: bool = True,
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Compute Singular Value Decomposition.

    Parameters
    ----------
    df: pd.DataFrame
        Matrix to decompose.
    svd_flip: bool
        Whether to use svd_flip on U and V or not.
    algorithm : str
        'arpack' or 'randomized'. Only applicable if approximate==True.
        See TruncatedSVD doc.

    Returns
    -------
    U: np.ndarray
        Unitary matrix having left singular vectors as columns.
    s: np.ndarray
        The singular values.
    V: np.ndarray
        Unitary matrix having right singular vectors as rows.
    """

    print(f'svd algorithm {algorithm}')
    print(f'svd n_components {n_components}')
    print(f'svd approximate {approximate}')

    if not approximate:
        U, s, V = linalg.svd(df, full_matrices=False)
        if svd_flip:
            U, V = extmath.svd_flip(U, V)

        return U, s, V

    algorithm_values = {'arpack', 'randomized'}

    if algorithm not in algorithm_values:
        raise ValueError(f"Algorithm must be in {algorithm_values}")


    X = csr_matrix(df)
    svd = TruncatedSVD(n_components=n_components, algorithm=algorithm, n_iter=7, random_state=42)
    svd.fit(X)

    print(f"SVD : Explained variance : {svd.explained_variance_}")
    print(f"SVD : Explained variance ratio: {svd.explained_variance_ratio_}")
    print(f"SVD : Explained variance ratio cumsum: {np.cumsum(svd.explained_variance_ratio_)}")
    
    U=svd.transform(X)
    V=svd.components_
    s=svd.singular_values_ 

    return U, s, V
