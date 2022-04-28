import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from saiph.reduction.utils.common import row_division, row_multiplication


def test_row_multiplication() -> None:
    df = pd.DataFrame([[1, 10], [2, 20]])
    expected = pd.DataFrame([[1, 10], [4, 40]])
    result = row_multiplication(df, np.array([1, 2]))

    assert_frame_equal(result, expected)


def test_row_division() -> None:
    df = pd.DataFrame([[1, 10], [2, 20]], dtype=float)
    expected = pd.DataFrame([[1, 10], [1, 10]], dtype=float)
    result = row_division(df, np.array([1, 2]))

    assert_frame_equal(result, expected)
