from typing import Tuple

from numpy.typing import ArrayLike
from scipy import linalg
from sklearn.utils import extmath

from saiph.models import DFLike


# Technically it returns three ndarray, which is a generic type
def SVD(df: DFLike, svd_flip=True) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Compute Singular Value Decomposition.

    Args:
        df: matrix to decompose

    Returns:
        U: unitary matrix having left singular vectors as columns
        s: the singular values
        V: unitary matrix having right singular vectors as rows
    """
    U, s, V = linalg.svd(df, full_matrices=False)
    if svd_flip:
        U, V = extmath.svd_flip(U, V)
    return U, s, V
