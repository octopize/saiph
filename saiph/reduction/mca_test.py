import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from saiph.reduction import DUMMIES_PREFIX_SEP
from saiph.reduction.mca import fit_transform, transform


def test_fit() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "hammer"],
            "score": ["aa", "aa"],
        }
    )

    result, model = fit_transform(df)

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

    assert_allclose(model.V, expected_v, atol=0.01)
    assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    assert_allclose(model.explained_var_ratio, expected_explained_var_ratio, atol=0.01),
    print(model._modalities)
    assert np.array_equal(
        model._modalities,
        [
            f"tool{DUMMIES_PREFIX_SEP}hammer",
            f"tool{DUMMIES_PREFIX_SEP}toaster",
            f"score{DUMMIES_PREFIX_SEP}aa",
        ],
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

    result, model = fit_transform(df)

    expected_result = pd.DataFrame(
        {
            "Dim. 1": [0.7, 0.7],
            "Dim. 2": [0.7, 0.7],
        }
    )
    expected_v = np.array([[1.0, 0.0], [0.0, 1.0]])
    expected_explained_var = np.array([0.0, 0.0])

    assert_frame_equal(result, expected_result, check_exact=False, atol=0.01)
    assert_allclose(model.V, expected_v, atol=0.01)
    assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    assert pd.isna(model.explained_var_ratio)
    assert np.array_equal(
        model._modalities,
        [f"tool{DUMMIES_PREFIX_SEP}toaster", f"score{DUMMIES_PREFIX_SEP}aa"],
    )
    assert_allclose(
        model.D_c,
        np.array([[1.414214, 0.0], [0.0, 1.414214]]),
        atol=0.01,
    )
    assert model.mean is None
    assert model.std is None


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

    result1, model1 = fit_transform(df)
    result2, model2 = fit_transform(df_2)

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


def test_transform_simple() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "toaster"],
            "score": ["aa", "aa"],
        }
    )
    _, model = fit_transform(df)

    df_transformed = transform(df, model)

    expected_transform = pd.DataFrame(
        {
            "Dim. 1": [0.707107, 0.707107],
            "Dim. 2": [0.707107, 0.707107],
        }
    )

    assert_frame_equal(
        df_transformed, expected_transform, check_exact=False, atol=0.00001
    )


def test_transform_vs_coord() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "toaster"],
            "score": ["aa", "aa"],
        }
    )
    coord, model = fit_transform(df)
    df_transformed = transform(df, model)

    assert_frame_equal(coord, df_transformed)
