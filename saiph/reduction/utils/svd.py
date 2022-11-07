from typing import Any, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import linalg
from sklearn.utils import extmath


def SVD(
    df: pd.DataFrame,
    nf:int,
    svd_flip: bool = True,
    seed:int=None
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
    if nf != min(pd.get_dummies(df).shape):
        # Randomized SVD
        U, s, V = direct_svd(df, q=2, l=nf, seed=seed)

    else:
        # Full SVD
        U, s, V = linalg.svd(df, full_matrices=False)

    if svd_flip:
        U, V = extmath.svd_flip(U, V)
  
    return U, s, V



def randomized_subspace_iteration(A, q:int=1, l:int=None, seed:int=None):
    """Algorithm 4.4 page 27
    q is the exponent of the power method
    l the target number of retained dimensions
    A is the input matrix 
    Q the matrix whose range approximates the range of A

    A: m x n
    Q: m x l
    """
    m, n = A.shape
    np.random.seed(seed)
    Omega = np.random.normal(loc=0, scale=1, size=(n, l))

    # Initialization
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y)

    # Iteration
    for i in range(q):
        Ytilde = A.transpose() @ Q
        Qtilde, _ = np.linalg.qr(Ytilde)
        Y = A @ Qtilde
        Q, _ = np.linalg.qr(Y)
    return Q


def direct_svd(A, q:int=1, l:int=None, seed:int=None):
    """Algorithm 5.1 page 29
    q is the exponent of the power method
    l the target number of retained dimensions
    A is the input matrix 
    Q the matrix whose range approximates the range of A

    A: (m, n)
    Q: (m, l)

    U: (m, l)
    S: (l,)
    V: (l, n)
    """
    if A.shape[1] > A.shape[0]:
        A = A.transpose()
        is_transposed = True
    else:
        is_transposed = False

    Q = randomized_subspace_iteration(A, q=q, l=l, seed=seed)

    B = Q.transpose() @ A
    Utilde, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Utilde

    if is_transposed:
        U_bis = U
        Vt_bis = Vt

        U = Vt_bis.transpose()
        Vt = U_bis.transpose()

    return U, S, Vt