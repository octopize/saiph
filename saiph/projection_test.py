import pandas as pd
import pytest

from saiph import fit


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.read_csv("fixtures/iris.csv")


def test_fit(df: pd.DataFrame) -> None:
    coord, model, param = fit(df, nf="all")
