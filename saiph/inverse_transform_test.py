from typing import List

import numpy as np
import pandas as pd
import pytest
from doubles import expect
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

import saiph
from saiph import projection

from saiph.inverse_transform import get_random_weighted_columns, inverse_transform


@pytest.mark.parametrize(
    "weights, expected_index",
    [
        ([0.3, 0.7, 0.01], 1),
        ([0.7, 0.3, 0.01], 0),
        ([0.01, 0.7, 0.3], 1),
        ([0.01, 0.3, 0.7], 2),
    ],
)
def test_get_random_weighted_columns(weights: List[float], expected_index: int) -> None:
    """Verify that get_random_weighted_columns returns the correct column."""
    df = pd.DataFrame(data=[weights])
    result = get_random_weighted_columns(df, np.random.default_rng(1))
    assert result.values[0] == expected_index


# TODO: add  def test_undummify()-> None: 
