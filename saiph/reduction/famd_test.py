import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas._testing.asserters import assert_series_equal
from pandas.testing import assert_frame_equal

from saiph.reduction.famd import fit


def test_fit_mix() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "hammer"],
            "score": ["aa", "ab"],
            "size": [1.0, 4.0],
        }
    )

    result, model, _ = fit(df)

    expected_result = pd.DataFrame(
        {
            "Dim. 1": [-1.73, 1.73],
            "Dim. 2": [0.0, 0.0],
        }
    )
    expected_v = np.array(
        [
            [0.57735, 0.408248, -0.408248, -0.408248, 0.408248],
            [0.816497, -0.288675, 0.288675, 0.288675, -0.288675],
        ]
    )
    expected_s = np.array([1.224745e00, 0.0])
    expected_u = np.array([[-1.0, 1.0], [1.0, 1.0]])
    expected_explained_var = np.array([1.5, 0.0])
    expected_explained_var_ratio = np.array([1.0, 0.0])

    assert_frame_equal(result, expected_result, check_exact=False, atol=0.01)
    assert_frame_equal(model.df, df)
    assert_allclose(model.V, expected_v, atol=0.01)
    assert_allclose(model.s, expected_s, atol=0.01)
    assert_allclose(model.U, expected_u, atol=0.01)
    assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    assert_allclose(model.explained_var_ratio, expected_explained_var_ratio, atol=0.01),
    # TODO: np.array_equal ?
    assert_allclose(model.variable_coord, model.V.T, atol=0.01)
    assert np.array_equal(
        model._modalities, ["tool_hammer", "tool_toaster", "score_aa", "score_ab"]
    )
    # Pertinent ?
    # assert_allclose(model.D_c,np.array([[2.0, 0.0, 0.0],
    # [0.0, 2.0, 0.0], [0.0, 0.0, 1.41421356]]), atol=0.01)
    assert model.D_c is None

    assert_allclose(model.mean, np.array(2.5))
    assert_allclose(model.std, np.array(1.5))

    assert_series_equal(
        model.prop.reset_index(drop=True),
        pd.Series([0.5, 0.5, 0.5, 0.5]).reset_index(drop=True),
        atol=0.01,
    )
    print(model._modalities)
    assert np.array_equal(
        model._modalities, ["tool_hammer", "tool_toaster", "score_aa", "score_ab"]
    )


def test_fit() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "hammer"],
            "score": ["aa", "aa"],
        }
    )

    result, model, _ = fit(df)

    expected = pd.DataFrame(
        {
            "Dim. 1": [1.0, -1.0],
            "Dim. 2": [0.0, 0.0],
        }
    )
    assert_frame_equal(result, expected, check_exact=False, atol=0.01)


def test_fit_zero() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "toaster"],
            "score": ["aa", "aa"],
        }
    )

    result, _, _ = fit(df)

    expected = pd.DataFrame(
        {
            "Dim. 1": [0.0, 0.0],
            "Dim. 2": [0.0, 0.0],
        }
    )
    assert_frame_equal(result, expected, check_exact=False, atol=0.01)
