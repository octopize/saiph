import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose

from saiph.pca import fit


def test_fit() -> None:
    df = pd.DataFrame(
        {
            "one": [1.0, 3.0],
            "two": [2.0, 4.0],
        }
    )

    result, model, _ = fit(df)

    expected = pd.DataFrame(
        {
            "Dim. 0": [-1.41, 1.41],
            "Dim. 1": [0.0, 0.0],
        }
    )
    assert_frame_equal(result, expected, check_exact=False, atol=0.01)

    expected_v = np.array([[0.71, 0.71], [-0.71, 0.71]])
    assert_allclose(model.V, expected_v, atol=0.01)


def test_fit_2() -> None:
    df = pd.DataFrame(
        {
            "one": [1.0, 2.0, 3.0],
            "two": [2.0, 4.0, 5.0],
        }
    )

    result, _, _ = fit(df)

    expected = pd.DataFrame(
        {
            "Dim. 0": [1.94, -0.28, -1.66],
            "Dim. 1": [-0.07, 0.18, -0.11],
        }
    )
    assert_frame_equal(result, expected, check_exact=False, atol=0.01)
