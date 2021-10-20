import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas._testing.asserters import assert_series_equal
from pandas.testing import assert_frame_equal

from saiph.reduction.famd import fit, center, scaler
from saiph.reduction.pca import fit as fit_pca
from saiph.reduction.pca import center as center_pca
from saiph.reduction.pca import scaler as scaler_pca
from saiph.reduction.mca import fit as fit_mca
from saiph.reduction.mca import center as center_mca
from saiph.reduction.mca import scaler as scaler_mca


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
    assert_allclose(model.variable_coord, model.V.T)
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


# TODO: Does not end (CENTER TO BE CHANGED - SHOULD BE OK)
# def test_fit_pca() -> None:
#     df = pd.DataFrame(
#         {
#             "one": [1.0, 3.0],
#             "two": [2.0, 4.0],
#         }
#     )

#     result, model, _ = fit(df)
#     result2, model2, _ = fit_pca(df)

    # assert_frame_equal(result, result2, check_exact=False, atol=0.01)

    # assert_frame_equal(model.df, df)
    # assert_allclose(model.V, expected_v, atol=0.01)
    # assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    # assert_allclose(model.explained_var_ratio, expected_explained_var_ratio, atol=0.01)
    # assert_allclose(model.variable_coord, model.V.T, atol=0.01)
    # assert_allclose(model.mean, [2.0, 3.0])
    # assert_allclose(model.std, [1.0, 1.0])


def test_fit_mca() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "hammer"],
            "score": ["aa", "aa"],
        }
    )

    result, model, _ = fit(df)
    result2, model2, _ = fit_mca(df)

    expected = pd.DataFrame(
        {
            "Dim. 1": [1.0, -1.0],
            "Dim. 2": [0.0, 0.0],
        }
    )

    assert_frame_equal(result, expected, check_exact=False, atol=0.01)

    # TODO: DOES NOT WORK !!
    # assert_frame_equal(result, result2)


def test_fit_zero() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "toaster"],
            "score": ["aa", "aa"],
        }
    )

    result, model, _ = fit(df)

    expected = pd.DataFrame(
        {
            "Dim. 1": [0.0, 0.0],
            "Dim. 2": [0.0, 0.0],
        }
    )
    assert_frame_equal(result, expected, check_exact=False, atol=0.01)



# TODO:
# def test_center_pca_mca() -> None:
#     df = pd.DataFrame(
#         {
#             "tool": ["toaster", "hammer"],
#             "score": ["aa", "ab"],
#             "size": [1.0, 4.0],
#             "age": [55, 62]
#         }
#     )

#     quali=["tool", "score"]
#     quanti=["size", "age"]
#     df_array1, mean1, std1, prop1, _modalities1 = center(df.copy(), quali=quali, quanti=quanti)
#     df2, mean2, std2 = center_pca(df[quanti].copy())
#     df_scale3, _modalities3, r3, c3 = center_mca(df[quali].copy())

#     print(df_array1)
#     print(df2)
#     print(df_scale3)

#     assert False

    # return df, mean, std
    # return df_scale, _modalities, r, c
    # return df_array, mean, std, prop, _modalities

# TODO:
# def test_center_pca_mca() -> None:
#     df = pd.DataFrame(
#         {
#             "tool": ["toaster", "hammer"],
#             "score": ["aa", "ab"],
#             "size": [1.0, 4.0],
#             "age": [55, 62]
#         }
#     )

#     result, model, param = fit(df)

#     quali=["tool", "score"]
#     quanti=["size", "age"]
#     param.quanti = quanti
#     param.quali = quali

#     df1 = scaler(model, param)
#     model.df = df[quanti]
#     df2 = scaler_pca(model)
#     model.df=df[quali]
#     df3 = scaler_mca(model)

#     print(df1)
#     print(df2)
#     print(df3)

#     assert False