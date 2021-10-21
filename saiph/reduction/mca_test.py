import numpy as np
import pandas as pd

import prince
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from saiph.reduction.mca import center, scaler
from saiph.reduction.mca import fit, transform


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
    # D_c only used in MCA, not even FAMD. To be removed ????
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


# TODO
def test_center_scaler() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "toaster"],
            "score": ["aa", "aa"],
        }
    )

    _, model, _ = fit(df, scale=True)

    print("original")
    print(model.df)
    print("Center")
    df1, modalities, r, c = center(model.df.copy())
    print(df1)
    print("type1")
    print(df1.dtypes)
    # print(mean)
    # print(std)
    print("scaler")
    df2 = scaler(model, None)
    print(df2)

    assert_frame_equal(
        df1, df2, check_column_type=False, check_names=False
    )

    assert False


# TODO; Gotta remove that, prince raises warnings
# STILL, it shows fit returns the right coord when vectors are colinear!!
# There is a sign problem next .... To be continued
def test_compare_prince_colin() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "toaster", "toaster"],
            "score": ["aa", "aa", "aa"],
            "car": ["tesla", "tesla", "tesla"]

        }
    )
    mca = prince.MCA(n_components=4)
    mca = mca.fit(df)
    mca = mca.transform(df)

    coord, _, _ = fit(df, scale=False)

    print(coord.to_numpy())
    print(mca)
    assert_allclose(coord.to_numpy(), mca, atol=0.0001)


# def test_compare_prince_general() -> None:
#     df = pd.DataFrame(
#         {
#             "tool": ["toaster", "toaster", "hammer"],
#             "score": ["aa", "ca", "bb"],
#             "car": ["tesla", "renault", "tesla"]
#         }
#     )
#     coord, model, param = fit(df, scale=False)
#     transf = transform(df, model, param)

#     mca = prince.MCA(n_components=4)
#     mca = mca.fit(df)
#     mca = mca.transform(df)


#     print(coord.to_numpy())
#     print(transf)
#     print(mca)
#     assert_allclose(coord.to_numpy(), mca, atol=0.0001)
#     assert False
