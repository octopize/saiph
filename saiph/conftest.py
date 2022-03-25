from typing import Any

import numpy as np
import pandas as pd
import pytest

from saiph.models import Model

_iris_csv = pd.read_csv("fixtures/iris.csv")


@pytest.fixture
def iris_df() -> pd.DataFrame:
    return _iris_csv


@pytest.fixture
def iris_quanti_df() -> pd.DataFrame:
    return _iris_csv.drop("variety", axis=1)


@pytest.fixture()
def quanti_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variable_1": [4, 5, 6, 7],
            "variable_2": [10, 20, 30, 40],
            "variable_3": [100, 50, -30, -50],
        }
    )


@pytest.fixture()
def quali_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tool": ["wrench", "wrench", "hammer", "hammer"],
            "fruit": ["apple", "orange", "apple", "apple"],
        }
    )


@pytest.fixture
def mixed_df():
    return pd.DataFrame(
        {
            "variable_1": [4, 5, 6, 7],
            "tool": ["wrench", "wrench", "hammer", "hammer"],
        }
    )


def check_model_equality(
    test: Model,
    expected: Model,
) -> None:
    """Verify that two Model instances are the same."""
    for key, value in expected.__dict__.items():
        test_item = test.__dict__[key]
        expected_item = value
        check_equality(test_item, expected_item)


def check_equality(
    test: Any,
    expected: Any,
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
