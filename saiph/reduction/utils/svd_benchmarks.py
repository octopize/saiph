from memory_profiler import profile
import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import NDArray

import pytest

from saiph.reduction.utils.svd import get_direct_randomized_svd

# To run this test, run:
# poetry run pytest -rP lsg/tests/randomized_svd_bench.py

# To profile memory with graph: 
# poetry run mprof run pytest -rP saiph/reduction/utils/svd_benchmarks.py
# poetry run mprof plot 


# Matrix to decompose
@pytest.fixture
def matrix() -> NDArray[np.float_]:
    """Generate a low rank matrix A = v @ v.T
    v is size (n, k)
    A is size (n, n) and rank k"""
    v = np.random.randint(low=0, high=100, size=(1000,10))
    A = v @ v.T
    return A

@profile
def test_full_svd(matrix)-> None:
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False) # Should be scipy 

    matrix_reconstruct = U @ np.diag(S) @ Vt

    assert_allclose(matrix, matrix_reconstruct, rtol=1e-3, atol=1e-6)

@profile
def test_randomized_svd(matrix)-> None:
    # l_retained_dimensions should be equal to the rank of the matrix
    U, S, Vt = get_direct_randomized_svd(matrix, l_retained_dimensions=10)

    matrix_reconstruct = U @ np.diag(S) @ Vt

    assert_allclose(matrix, matrix_reconstruct, rtol=1e-3, atol=1e-6)






