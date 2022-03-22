import pandas as pd
import pytest

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
            "variable_1": [4, 5, 6, 7, 11, 2, 52],
            "variable_2": [10, 20, 30, 40, 10, 74, 10],
            "variable_3": [100, 50, -30, -50, -19, -29, -20],
        }
    )


@pytest.fixture()
def quali_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tool": [
                "toaster",
                "toaster",
                "hammer",
                "toaster",
                "toaster",
                "hammer",
                "toaster",
                "toaster",
                "hammer",
            ],
            "score": ["aa", "ca", "bb", "aa", "ca", "bb", "aa", "ca", "bb"],
            "car": [
                "tesla",
                "renault",
                "tesla",
                "tesla",
                "renault",
                "tesla",
                "tesla",
                "renault",
                "tesla",
            ],
            "moto": [
                "Bike",
                "Bike",
                "Motor",
                "Bike",
                "Bike",
                "Motor",
                "Bike",
                "Bike",
                "Motor",
            ],
        }
    )


@pytest.fixture
def mixed_df():
    return pd.DataFrame(
        {
            "variable_1": [4, 5, 6, 7, 11, 2, 52],
            "variable_2": [10, 20, 30, 40, 10, 74, 10],
            "variable_3": ["red", "blue", "blue", "green", "red", "blue", "red"],
            "variable_4": [100, 50, -30, -50, -19, -29, -20],
        }
    )
