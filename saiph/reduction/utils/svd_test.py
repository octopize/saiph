import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_almost_equal
from saiph.reduction.utils.svd import get_svd, get_randomized_subspace_iteration, get_direct_randomized_svd


# Matrix to decompose
@pytest.fixture
def test_matrix():
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    A = pd.DataFrame(A)
    return A

def test_full_svd(test_matrix: pd.DataFrame) -> None:
    expected_U = np.array([[ 0.21483724,  0.88723069, -0.40824829],
       [ 0.52058739,  0.24964395,  0.81649658],
       [ 0.82633754, -0.38794278, -0.40824829]])

    expected_S = np.array([1.68481034e+01, 1.06836951e+00, 3.33475287e-16])

    expected_Vt = np.array([[ 0.47967118,  0.57236779,  0.66506441],
       [-0.77669099, -0.07568647,  0.62531805],
       [ 0.40824829, -0.81649658,  0.40824829]])

    # Should return a full SVD using scipy implementation
    U, S, Vt = get_svd(test_matrix, nf=np.min(pd.get_dummies(test_matrix).shape), seed=2)

    assert_array_almost_equal(U, expected_U, decimal=6)
    assert_array_almost_equal(S, expected_S, decimal=6)
    assert_array_almost_equal(Vt, expected_Vt, decimal=6)


def test_randomized_subspace(test_matrix: pd.DataFrame) -> None:
    expected_Q = np.array(
        [[-0.21483746,  0.88723063],
        [-0.52058745,  0.24964382],
        [-0.82633744, -0.38794299]]
    )

    Q = get_randomized_subspace_iteration(test_matrix, q=2, l=2, seed=2)

    assert_array_almost_equal(Q, expected_Q, decimal=6)


def test_direct_svd(test_matrix: pd.DataFrame) -> None:

    expected_U = np.array([[ 0.21483724,  0.88723069],
        [ 0.52058739,  0.24964395],
        [ 0.82633754, -0.38794278]]
    )

    expected_S = np.array([16.84810335,  1.06836951])

    expected_Vt = np.array([[ 0.47967118,  0.57236779,  0.66506441],
       [-0.77669099, -0.07568647,  0.62531805]])

    U, S, Vt = get_direct_randomized_svd(test_matrix, q=2, l=2, seed=2)

    assert_array_almost_equal(U, expected_U, decimal=6)
    assert_array_almost_equal(S, expected_S, decimal=6)
    assert_array_almost_equal(Vt, expected_Vt, decimal=6)