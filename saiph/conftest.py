import pandas as pd
import pytest

_iris_csv = pd.read_csv("fixtures/iris.csv")
_wbcd_csv = pd.read_csv("fixtures/breast_cancer_wisconsin.csv").drop(columns='Sample_code_number')


@pytest.fixture
def iris_df() -> pd.DataFrame:
    return _iris_csv


@pytest.fixture
def iris_quanti_df() -> pd.DataFrame:
    return _iris_csv.drop("variety", axis=1)

@pytest.fixture
def wbcd_df() -> pd.DataFrame:
    return _wbcd_csv