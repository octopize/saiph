import pandas as pd
import pytest

from saiph import fit
from pandas.testing import assert_frame_equal


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.read_csv("fixtures/iris.csv")


def test_fit(df: pd.DataFrame) -> None:
    coord, model, param = fit(df, nf="all")


def test_fit_numeric() -> None:
    df = pd.DataFrame(
        {
            "one": [1.0, 3],
            "two": [2.0, 4],
        }
    )

    result, _, _ = fit(df)

    expected = pd.DataFrame(
        {
            "Dim. 0": [-1.41, 1.41],
            "Dim. 1": [0.0, 0.0],
        }
    )
    assert_frame_equal(result, expected, check_exact=False, atol=0.01)
