import pandas as pd
from pandas.testing import assert_frame_equal

from saiph import fit, inverse_transform, transform

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

def test_transform_then_inverse_FAMD_weight() -> None:
    df = pd.DataFrame(
        {
            'variable_1': [4, 5, 6, 7, 11, 2, 52],
            'variable_2': [10, 20, 30, 40, 10, 74, 10],
            'variable_3': [100, 50, -30, -50, -19, -29, -20],
            'variable_4': ['red', 'blue', 'blue', 'green', 'red', 'blue', 'red']
    }
    )

    _, model, param = fit(df, col_w= [2, 1, 3, 2])
    transformed = transform(df, model, param)
    un_transformed = inverse_transform(transformed, model, param)

    assert_frame_equal(un_transformed, df)


def test_transform_then_inverse_ACP_weight() -> None:
    df = pd.DataFrame(
        {
            'variable_1': [4, 5, 6, 7, 11, 2, 52],
            'variable_2': [10, 20, 30, 40, 10, 74, 10],
            'variable_3': [100, 50, -30, -50, -19, -29, -20],
            
    }
    )

    _, model, param = fit(df, col_w= [2, 1, 3])
    transformed = transform(df, model, param)
    un_transformed = inverse_transform(transformed, model, param)

    assert_frame_equal(un_transformed, df)