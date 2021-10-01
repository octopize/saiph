import pytest
import pandas as pd

from saiph import fit


@pytest.fixture
def df():
    return pd.read_csv('fixtures/iris.csv')


def test_fit(df):
    coord, model, param = fit(df, nf = 'all')
