import pandas as pd
from pandas.testing import assert_frame_equal

from saiph import fit


def test_fit(iris_df: pd.DataFrame) -> None:
    coord, model, param = fit(iris_df, nf="all")
    print(coord)
    # print(model)
    print(param)
    # assert False


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
            "Dim. 1": [-1.41, 1.41],
            "Dim. 2": [0.0, 0.0],
        }
    )
    assert_frame_equal(result, expected, check_exact=False, atol=0.01)
