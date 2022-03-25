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
        pd.Series([1.2], index=["Row. 1"]),
        np.array([[1.2], [1.3]]),
    ],
)
def test_encode_decode_single_items(item: Any) -> None:
    """Verify that we encode dataframes and arrays separately."""
    encoded = json.dumps(item, cls=NumpyPandasEncoder)
    decoded = json.loads(encoded, object_hook=numpy_pandas_json_obj_hook)
    check_equality(item, decoded)


def test_encode_decode_model(mixed_df: pd.DataFrame):
    """Verify that we can encode and decode a model."""
    expected_model = fit(mixed_df)
    raw_model = ModelJSONSerializer.dumps(expected_model)
    decoded_model = ModelJSONSerializer.loads(raw_model)

    check_model_equality(decoded_model, expected_model)
