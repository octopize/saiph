# type: ignore
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from saiph.conftest import check_equality, check_model_equality
from saiph.projection import fit
from saiph.serializer import (
    ModelJSONSerializer,
    NumpyPandasEncoder,
    numpy_pandas_json_obj_hook,
)


@pytest.mark.parametrize(
    "item",
    [
        pd.DataFrame([[1.2]], columns=["col 1"], index=[2]),
        pd.DataFrame([[1.2]], columns=["col 1"], index=["2"]),
        pd.DataFrame([[1.2]], columns=[0], index=[2]),
        pd.DataFrame([[1.2]], columns=["0"], index=[2]),
        pd.DataFrame([[1]]),
        pd.DataFrame([["1"]]),
        pd.DataFrame([["1", np.nan]]),
        pd.DataFrame([[np.nan, "1"]]),
        pd.DataFrame([[np.nan, 1]]),
        pd.DataFrame([[np.nan, np.nan]]),
        pd.Series([1]),
        pd.Series(["1"]),
        pd.Series(["1", np.nan]),
        pd.Series([np.nan, "1"]),
        pd.Series([np.nan, 1]),
        pd.Series([np.nan, np.nan]),
        pd.Series([1.2], index=["Row. 1"]),
        pd.Series([1.3], index=[0]),
        pd.Series([1.4], index=["0"]),
        np.array([[1.2], [1.3]]),
        np.array([[1], [1]]),
        np.array([[np.nan], [1.3]]),
        np.array([[np.nan], [1]]),
        np.array([[np.nan], [np.nan]]),
    ],
)
def test_encode_decode_single_items(item: Any) -> None:
    """Verify that we encode dataframes and arrays separately."""
    encoded = json.dumps(item, cls=NumpyPandasEncoder)
    decoded = json.loads(encoded, object_hook=numpy_pandas_json_obj_hook)
    check_equality(decoded, item)


def test_encode_decode_model(mixed_df: pd.DataFrame) -> None:
    """Verify that we can encode and decode a model."""
    expected_model = fit(mixed_df)
    raw_model = ModelJSONSerializer.dumps(expected_model)
    decoded_model = ModelJSONSerializer.loads(raw_model)

    check_model_equality(decoded_model, expected_model)
