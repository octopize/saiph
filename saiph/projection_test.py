import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from saiph import fit, inverse_transform, transform, stats

# mypy: ignore-errors


def test_transform_then_inverse_FAMD(iris_df: pd.DataFrame) -> None:
    _, model, param = fit(iris_df, nf="all")
    transformed = transform(iris_df, model, param)
    un_transformed = inverse_transform(transformed, model, param)

    print(un_transformed)
    print(iris_df)

    assert_frame_equal(un_transformed, iris_df)


def test_transform_then_inverse_PCA(iris_quanti_df: pd.DataFrame) -> None:
    _, model, param = fit(iris_quanti_df, nf="all")
    transformed = transform(iris_quanti_df, model, param)
    un_transformed = inverse_transform(transformed, model, param)

    assert_frame_equal(un_transformed, iris_quanti_df)


def test_transform_then_inverse_MCA() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "toaster", "hammer"],
            "score": ["aa", "ca", "bb"],
            "car": ["tesla", "renault", "tesla"],
            "moto": ["Bike", "Bike", "Motor"],
        }
    )

    _, model, param = fit(df)
    transformed = transform(df, model, param)
    un_transformed = inverse_transform(transformed, model, param)

    assert_frame_equal(un_transformed, df)


def test_transform_then_inverse_MCA_type() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "toaster", "hammer"],
            "score": [1, 1, 0],
            "car": ["tesla", "renault", "tesla"],
            "moto": ["Bike", "Bike", "Motor"],
        }
    )

    df = df.astype("object")
    _, model, param = fit(df)
    transformed = transform(df, model, param)
    un_transformed = inverse_transform(transformed, model, param)

    assert_frame_equal(un_transformed, df)


def test_transform_then_inverse_FAMD_weighted() -> None:
    df = pd.DataFrame(
        {
            "variable_1": [4, 5, 6, 7, 11, 2, 52],
            "variable_2": [10, 20, 30, 40, 10, 74, 10],
            "variable_3": ["red", "blue", "blue", "green", "red", "blue", "red"],
            "variable_4": [100, 50, -30, -50, -19, -29, -20],
        }
    )

    _, model, param = fit(df, col_w=[2, 1, 3, 2])
    transformed = transform(df, model, param)
    un_transformed = inverse_transform(transformed, model, param)

    assert_frame_equal(un_transformed, df)


def test_transform_then_inverse_PCA_weighted() -> None:
    df = pd.DataFrame(
        {
            "variable_1": [4, 5, 6, 7, 11, 2, 52],
            "variable_2": [10, 20, 30, 40, 10, 74, 10],
            "variable_3": [100, 50, -30, -50, -19, -29, -20],
        }
    )

    _, model, param = fit(df, col_w=[2, 1, 3])
    transformed = transform(df, model, param)
    un_transformed = inverse_transform(transformed, model, param)

    assert_frame_equal(un_transformed, df)


def test_transform_then_inverse_MCA_weighted() -> None:
    df = pd.DataFrame(
        {
            "variable_1": ["1", "3", "3", "3", "1", "2", "2", "1", "1", "2"],
            "variable_2": ["1", "1", "1", "2", "2", "1", "1", "1", "2", "2"],
            "variable_3": ["1", "2", "1", "2", "1", "2", "1", "1", "2", "2"],
            "variable_4": [
                "red",
                "blue",
                "blue",
                "green",
                "red",
                "blue",
                "red",
                "red",
                "red",
                "red",
            ],
        }
    )

    _, model, param = fit(df, col_w=[2, 1, 3, 2])
    transformed = transform(df, model, param)
    un_transformed = inverse_transform(transformed, model, param)

    assert_frame_equal(un_transformed, df)


def test_coords_vs_transform_with_multiple_nf(iris_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        fit(iris_df, nf=10000)
    
    with pytest.raises(ValueError):
        fit(iris_df, nf=-1)
    
    for n in range(7):
        coord, model, param = fit(iris_df, nf=n)
        transformed = transform(iris_df, model, param)
        assert_frame_equal(coord, transformed)


df_pca = pd.DataFrame(
    {
        0: [1000.0, 3000.0, 10000.0, 1500.0, 700.0, 3300.0, 5000.0, 2000.0, 1200.0, 6000.0],
        1: [185.0, 174.3, 156.8, 182.7, 180.3, 179.2, 164.7, 192.5, 191.0, 169.2],
        2: [1, 5, 10 ,2, 4, 4, 7, 3, 1, 6]
    }
)

df_famd = pd.DataFrame(
    {
        "variable_1": [4, 5, 6, 7, 11, 2, 52],
        "variable_2": [10, 20, 30, 40, 10, 74, 10],
        "variable_3": ["red", "blue", "blue", "green", "red", "blue", "red"],
        "variable_4": [100, 50, -30, -50, -19, -29, -20],
    }
)

df_mca = pd.DataFrame(
    {
        0: ['red', 'red', 'whithe', 'whithe', 'red', 'whithe', 'red', 'red', 'whithe', 'red'],
        1: ['beef', 'chicken', 'fish', 'fish', 'beef', 'chicken', 'beef', 'chicken', 'fish', 'beef'],
        2: ['france', 'espagne', 'france' ,'italy', 'espagne', 'france', 'france', 'espagne', 'chine', 'france']
    }
)


@pytest.mark.parametrize(
    "df_input,expected_type", [(df_pca, "pca"), (df_mca, "mca"), (df_famd, "famd")]
)
def test_eval(df_input, expected_type):
    _, model, _ = fit(df_input)
    assert model.type == expected_type

expected_pca_contrib = [32.81178064277649, 33.12227570926467, 34.065943647958846]
expected_mca_contrib = [13.314231201732547, 19.971346802598834, 7.924987451762375, 2.8647115861394203, 24.435070805047193, 11.119305005676665, 8.778349565191498, 0.47269257617482885, 11.119305005676662]
expected_famd_contrib = [15.696161557629662, 36.08406414786589, 11.420291290785196, 9.852860955848701, 6.104123324745425, 20.842498723125182]

@pytest.mark.parametrize(
    "df_input,expected_contrib", [(df_pca, expected_pca_contrib), (df_mca, expected_mca_contrib), (df_famd, expected_famd_contrib)]
)
def test_var_contrib(df_input, expected_contrib):
    _, model, param = fit(df_input)
    stats(model, param)
    assert list(param.contrib['Dim. 1']) == expected_contrib