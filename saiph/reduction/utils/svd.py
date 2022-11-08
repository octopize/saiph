from typing import Any, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import linalg
from sklearn.utils import extmath


def get_svd(
    df: pd.DataFrame,
    nf:int,
    svd_flip: bool = True,
    seed:int=None
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Compute Singular Value Decomposition.

    Arguments
    ---------
        df: Matrix to decompose, shape (m, n)
        nf: target number of dimensions to retain (number of singular values)
        svd_flip: Whether to use svd_flip on U and V or not. Default `True`
        seed: random seed. Default `None`

    Returns
    -------
        U: unitary matrix having left singular vectors as columns, shape (m,l)
        S: vector of the singular values, shape (l,)
        Vt: unitary matrix having right singular vectors as rows, shape (l,n)
    """
    if nf != min(pd.get_dummies(df).shape):
        # Randomized SVD
        U, S, Vt = get_direct_randomized_svd(df, q=2, l=nf, seed=seed)

    else:
        # Full SVD
        U, S, Vt = linalg.svd(df, full_matrices=False)

    if svd_flip:
        U, Vt = extmath.svd_flip(U, Vt)
  
    return U, S, Vt



def get_randomized_subspace_iteration(A, l:int, q:int=2, seed:int=None):
    """Generate a subspace for more efficient SVD compuation using random methods.
    
    From http://arxiv.org/abs/0909.4061, algorithm 4.4 page 27

    Arguments
    ---------
        A: input matrix, shape (m, n)
        l: target number of retained dimensions, l<min(m,n)
        q: exponent of the power method. Higher this exponent, the more precise will be the SVD, but more complex to compute. Default `2`
        seed: random seed. Default `None`
        
    Returns
    -------
        Q: matrix whose range approximates the range of A, shape (m, l)
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


def get_direct_randomized_svd(A, l:int, q:int=2, seed:int=None):
    """Compute a fixed-rank SVD approximation using random methods.
    
    From http://arxiv.org/abs/0909.4061, algorithm 5.1 page 29

    Arguments
    ---------
        A: input matrix, shape (m, n)
        l: target number of retained dimensions, l<min(m,n)
        q: exponent of the power method. Higher this exponent, the more precise will be the SVD, but more complex to compute.
        seed: random seed. Default `None`

    Returns
    -------
        U: unitary matrix having left singular vectors as columns, shape (m,l)
        S: vector of the singular values, shape (l,)
        Vt: unitary matrix having right singular vectors as rows, shape (l,n)
    """
    if A.shape[1] > A.shape[0]:
        A = A.transpose()
        is_transposed = True
    else:
        is_transposed = False

    # Q: matrix whose range approximates the range of A, shape (m, l)
    Q = get_randomized_subspace_iteration(A, q=q, l=l, seed=seed)

    B = Q.transpose() @ A
    Utilde, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Utilde

    if is_transposed:
        U_bis = U
        Vt_bis = Vt

        U = Vt_bis.transpose()
        Vt = U_bis.transpose()

    return U, S, Vt