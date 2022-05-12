from typing import List

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from saiph import projection
from saiph.inverse_transform import get_random_weighted_columns, undummify


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


@pytest.mark.parametrize(
    "use_max_modalities, expected",
    [
        (
            True,
            pd.DataFrame(
                [["wrench", "orange"], ["hammer", "apple"]], columns=["tool", "fruit"]
            ),
        ),
        (
            False,
            pd.DataFrame(
                [["wrench", "orange"], ["wrench", "apple"]], columns=["tool", "fruit"]
            ),
        ),
    ],
)
def test_undummify(
    quali_df: pd.DataFrame, use_max_modalities: bool, expected: pd.DataFrame
) -> None:
    model = projection.fit(quali_df)

    dummy_df = pd.DataFrame(
        [[0.3, 0.7, 0.01, 0.99], [0.51, 0.49, 0.8, 0.2]],
        columns=["tool___hammer", "tool___wrench", "fruit___apple", "fruit___orange"],
    )
    df = undummify(dummy_df, model, use_max_modalities=use_max_modalities, seed=321)

    assert_frame_equal(df, expected)
