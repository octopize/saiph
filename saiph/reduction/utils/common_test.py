import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from saiph.reduction import DUMMIES_PREFIX_SEP
from saiph.reduction.utils.common import (
    column_multiplication,
    get_dummies_mapping,
    row_division,
    row_multiplication,
)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame([[1, 10], [2, 20]])


def test_row_multiplication(df: pd.DataFrame) -> None:
    expected = pd.DataFrame([[1, 10], [4, 40]])

    result = row_multiplication(df, np.array([1, 2]))

    assert_frame_equal(result, expected)


def test_column_multiplication(df: pd.DataFrame) -> None:
    expected = pd.DataFrame([[1, 20], [2, 40]])

    result = column_multiplication(df, np.array([1, 2]))
    assert_frame_equal(result, expected)


def test_row_division(df: pd.DataFrame) -> None:
    expected = pd.DataFrame([[1, 10], [1, 10]], dtype=float)

    result = row_division(df, np.array([1, 2]))

    assert_frame_equal(result, expected)


def test_get_dummies_mapping(quali_df: pd.DataFrame) -> None:
    dummy_columns = pd.get_dummies(quali_df, prefix_sep=DUMMIES_PREFIX_SEP).columns

    result = get_dummies_mapping(quali_df.columns, dummy_columns)

    sep = DUMMIES_PREFIX_SEP
    expected = {
        "tool": [f"tool{sep}hammer", f"tool{sep}wrench"],
        "fruit": [f"fruit{sep}apple", f"fruit{sep}orange"],
    }
    assert result == expected
