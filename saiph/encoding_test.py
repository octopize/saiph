import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from saiph.conftest import check_equality, check_model_equality
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
    check_equality(item, decoded)


def test_encode_decode_model(mixed_df: pd.DataFrame) -> None:
    """Verify that the actual fitted model can be encoded and decoded."""
    model = fit(mixed_df, col_w=np.array([2, 1, 3, 2]))

    encoded = json.dumps(model.__dict__, cls=ModelEncoder)
    decoded_dict = json.loads(encoded, object_hook=json_model_obj_hook)
    decoded_model = Model(**decoded_dict)

    check_model_equality(model, decoded_model)
