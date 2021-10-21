import pandas as pd
import pytest

_iris_csv = pd.read_csv("fixtures/iris.csv")


@pytest.fixture
def iris_df() -> pd.DataFrame:
    return _iris_csv


@pytest.fixture
def iris_quanti_df() -> pd.DataFrame:
    return _iris_csv.drop("variety", axis=1)
