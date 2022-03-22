import json
from typing import Any, Union

import numpy as np
import pandas as pd
import pytest

from saiph.encoding import ModelEncoder, json_model_obj_hook
from saiph.models import Model
from saiph.projection import fit


@pytest.mark.parametrize(
    "item",
    [
        pd.DataFrame([[1.2]], columns=["col 1"], index=[2]),
        pd.Series([1.2], index=["Row. 1"]),
        np.array([[1.2], [1.3]]),
    ],
)
def test_encode_decode_single_items(item: Any) -> None:
    """Verify that we encode dataframes and arrays separately."""
    encoded = json.dumps(item, cls=ModelEncoder)
    decoded = json.loads(encoded, object_hook=json_model_obj_hook)

    check_equality(decoded, item)


def test_encode_decode_model(mixed_df: pd.DataFrame) -> None:
    """Verify that the actual fitted model can be encoded and decoded."""
    _, model = fit(mixed_df, col_w=np.array([2, 1, 3, 2]))

    encoded = json.dumps(model.__dict__, cls=ModelEncoder)
    decoded_dict = json.loads(encoded, object_hook=json_model_obj_hook)
    decoded_model = Model(**decoded_dict)

    for key, value in model.__dict__.items():
        check_equality(value, decoded_model.__dict__[key])


def check_equality(
    test: Union[pd.Series, pd.DataFrame, np.ndarray],
    expected: Union[pd.Series, pd.DataFrame, np.ndarray],
) -> None:
    """Check equality of dataframes, series and np.arrays."""
    if isinstance(test, pd.DataFrame) and isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(test, expected)
    elif isinstance(test, pd.Series) and isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(test, expected)
    elif isinstance(test, np.ndarray) and isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(test, expected)
    else:
        assert test == expected
