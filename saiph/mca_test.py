import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from saiph.mca import fit


def test_fit() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "hammer"],
            "score": ["aa", "aa"],
        }
    )

    result, model, _ = fit(df)

    expected_result = pd.DataFrame(
        {
            "Dim. 1": [0.7, -0.7],
            "Dim. 2": [-0.7, -0.7],
        }
    )
    expected_v = np.array([[-0.707107, 0.707107, -0.0], [-0.707107, -0.707107, 0.0]])
    expected_explained_var = np.array([1.25000e-01, 3.85186e-34])
    expected_explained_var_ratio = np.array([1.0, 0.0])

    assert_frame_equal(result, expected_result, check_exact=False, atol=0.01)

    assert_frame_equal(model.df, df)
    assert_allclose(model.V, expected_v, atol=0.01)
    assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    assert_allclose(model.explained_var_ratio, expected_explained_var_ratio, atol=0.01),
    # TODO: Why is it different in MCA ???
    # D_c only used in MCA, not even FAMD. Tu remove ????
    # assert_allclose(model.variable_coord, model.V.T, atol=0.01)
    assert np.array_equal(
        model._modalities, ["tool_hammer", "tool_toaster", "score_aa"]
    )
    assert_allclose(
        model.D_c,
        np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.41421356]]),
        atol=0.01,
    )
    assert model.mean is None
    assert model.std is None


def test_fit_zero() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "toaster"],
            "score": ["aa", "aa"],
        }
    )

    result, model, _ = fit(df)

    expected_result = pd.DataFrame(
        {
            "Dim. 1": [0.7, 0.7],
            "Dim. 2": [0.7, 0.7],
        }
    )
    assert_frame_equal(result, expected_result, check_exact=False, atol=0.01)
    # TODO: complete that


def test_fit_zero_same_df() -> None:
    """Verify that we get the same result if the pattern matches."""
    df = pd.DataFrame(
        {
            "tool": ["toaster", "toaster"],
            "score": ["aa", "aa"],
        }
    )
    df_2 = pd.DataFrame(
        {
            "tool": ["hammer", "hammer"],
            "score": ["bb", "bb"],
        }
    )

    result1, model1, _ = fit(df)
    result2, model2, _ = fit(df_2)

    assert_frame_equal(result1, result2)

    for k in [
        "explained_var",
        "variable_coord",
        "variable_coord",
        "U",
        "s",
        "mean",
        "std",
        "prop",
        "D_c",
    ]:  # removed "_modalities", "df", "explained_var_ratio"
        k1 = getattr(model1, k)
        k2 = getattr(model2, k)
        if isinstance(k1, pd.DataFrame):
            assert k1.equals(k2)
        elif isinstance(k1, np.ndarray):
            assert np.array_equal(k1, k2)
        else:
            assert k1 == k2
