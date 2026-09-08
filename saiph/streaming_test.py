from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from pandas.testing import assert_frame_equal, assert_series_equal

from saiph.exception import InvalidParameterException
from saiph.inverse_transform import inverse_transform
from saiph.models import Model
from saiph.projection import transform
from saiph.reduction import DUMMIES_SEPARATOR, famd, mca, pca
from saiph.streaming import (
    DecompositionAccumulator,
    ScalingAccumulator,
    ScalingParams,
    fit_streaming,
)


def chunks(df: pd.DataFrame, size: int) -> Iterator[pd.DataFrame]:
    for start in range(0, len(df), size):
        yield df.iloc[start : start + size]


def numerical_rank(s: NDArray[np.float64], rtol: float = 1e-10) -> int:
    """Number of singular values that carry signal.

    FAMD and MCA are structurally rank-deficient, and the axes past the rank are an
    arbitrary basis of the null space in every implementation. Comparing them
    measures the tie-breaking of the decomposition, not the fit.
    """
    return int(np.sum(s > rtol * s[0]))


def align_signs(
    V: NDArray[np.float64], reference: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Flip each axis of `V` to point the same way as `reference`.

    A singular vector is defined up to sign, and a streaming fit has no left
    singular vectors to break the tie the same way `fit` does.
    """
    signs = np.where((V * reference).sum(axis=1) < 0, -1.0, 1.0)
    return V * signs[:, np.newaxis], reference


def assert_agrees_with_reference(
    streamed: Model, reference: Model, *, rtol: float = 1e-11, atol: float = 1e-10
) -> None:
    """Assert a streamed fit found the same model as the whole-table fit."""
    assert streamed.type == reference.type
    assert streamed.original_continuous == reference.original_continuous
    assert streamed.original_categorical == reference.original_categorical
    assert streamed.dummy_categorical == reference.dummy_categorical
    assert streamed.modalities_types == reference.modalities_types
    assert_allclose(streamed.column_weights, reference.column_weights)
    assert_allclose(streamed.row_weights, reference.row_weights)

    if reference.mean is not None:
        assert_series_equal(streamed.mean, reference.mean, rtol=rtol)
        assert_series_equal(streamed.std, reference.std, rtol=rtol)
    if reference.prop is not None:
        assert_series_equal(streamed.prop, reference.prop, rtol=rtol)
    if reference._modalities is not None:
        assert streamed._modalities is not None
        assert list(streamed._modalities) == list(reference._modalities)
    if reference.D_c is not None:
        assert streamed.D_c is not None
        assert_allclose(streamed.D_c, reference.D_c, rtol=rtol)
    if reference.dummies_col_prop is not None:
        assert streamed.dummies_col_prop is not None
        assert_allclose(streamed.dummies_col_prop, reference.dummies_col_prop, rtol=rtol)

    assert reference.s is not None and streamed.s is not None
    k = numerical_rank(reference.s)
    assert k > 0
    assert_allclose(streamed.s[:k], reference.s[:k], rtol=rtol)
    assert_allclose(streamed.explained_var[:k], reference.explained_var[:k], rtol=rtol)
    assert_allclose(streamed.explained_var_ratio[:k], reference.explained_var_ratio[:k], rtol=rtol)
    assert_allclose(*align_signs(streamed.V[:k], reference.V[:k]), atol=atol)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 3, 7, 64, 150, 400])
def test_fit_streaming_pca_equals_fit(iris_quanti_df: pd.DataFrame, size: int) -> None:
    df = iris_quanti_df
    nf = min(df.shape)

    reference = pca.fit(df, nf=nf)
    streamed = fit_streaming(lambda: chunks(df, size), nf=nf)

    assert_agrees_with_reference(streamed, reference)


def test_fit_streaming_pca_independent_of_batch_order(iris_quanti_df: pd.DataFrame) -> None:
    df = iris_quanti_df
    nf = min(df.shape)
    shuffled = df.sample(frac=1, random_state=0)

    reference = pca.fit(df, nf=nf)
    streamed = fit_streaming(lambda: chunks(shuffled, 11), nf=nf)

    assert_agrees_with_reference(streamed, reference)


def test_fit_streaming_pca_with_col_weights(iris_quanti_df: pd.DataFrame) -> None:
    df = iris_quanti_df
    nf = min(df.shape)
    col_weights = np.array([3.0, 1.0, 1.0, 2.0])

    reference = pca.fit(df, nf=nf, col_weights=col_weights)
    streamed = fit_streaming(lambda: chunks(df, 13), nf=nf, col_weights=col_weights)

    assert_agrees_with_reference(streamed, reference)


def test_fit_streaming_pca_with_constant_column() -> None:
    """A column with a null std is divided by a guarded 1, as `fit` does."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "varying": rng.normal(size=40),
            "constant": np.full(40, 3.0),
            "other": rng.normal(size=40),
        }
    )
    nf = 2

    reference = pca.fit(df, nf=min(df.shape))
    streamed = fit_streaming(lambda: chunks(df, 6), nf=nf)

    assert_agrees_with_reference(streamed, reference)


def test_fit_streaming_pca_with_null_continuous_value() -> None:
    """A null is skipped by the mean and std, as `fit` skips it."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=30), "b": rng.normal(size=30)})
    df.loc[3, "a"] = np.nan
    df.loc[17, "b"] = np.nan

    params = _scaling_params(df, size=7)

    assert_series_equal(params.mean, df.mean(), rtol=1e-12)
    assert_series_equal(params.std, df.std(ddof=0), rtol=1e-12, check_names=False)


@pytest.mark.parametrize(
    ("offset", "rtol"),
    [(1e6, 1e-11), (1e9, 1e-8)],
)
def test_fit_streaming_std_is_accurate_on_an_offset_column(offset: float, rtol: float) -> None:
    """The variance update must not be the raw-moment one.

    `fit` requires datetimes as seconds since epoch, so a column whose values sit
    around 1e9 with a small spread is an ordinary input, not a pathological one.
    Subtracting two large squares there gives a relative error above 1, while the
    incremental update stays near 1e-9.
    """
    rng = np.random.default_rng(2)
    values = offset + rng.normal(size=200_000)
    df = pd.DataFrame({"epoch_seconds": values})
    truth = float(np.std(values.astype(np.longdouble)))

    params = _scaling_params(df, size=1000)

    assert_allclose(params.std["epoch_seconds"], truth, rtol=rtol)

    n = len(values)
    mean = values.sum() / n
    raw_moment_std = np.sqrt(max((values**2).sum() / n - mean * mean, 0.0))
    assert abs(raw_moment_std - truth) / truth > 1e3 * rtol


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


def _scaling_params(df: pd.DataFrame, size: int) -> ScalingParams:
    scaling = ScalingAccumulator()
    for batch in chunks(df, size):
        scaling.partial_fit(batch)
    return scaling.finalize()


def test_fit_streaming_rejects_an_iterator(iris_quanti_df: pd.DataFrame) -> None:
    """An iterator is exhausted by pass 1, so pass 2 would fit on no rows at all."""
    with pytest.raises(InvalidParameterException, match="callable"):
        fit_streaming(chunks(iris_quanti_df, 10), nf=2)  # type: ignore[arg-type]


def test_fit_streaming_rejects_nf_above_the_rank(iris_quanti_df: pd.DataFrame) -> None:
    with pytest.raises(InvalidParameterException, match="0 < 'nf' <= 4"):
        fit_streaming(lambda: chunks(iris_quanti_df, 10), nf=5)


def test_fit_streaming_rejects_nf_of_zero(iris_quanti_df: pd.DataFrame) -> None:
    with pytest.raises(InvalidParameterException, match="0 < 'nf'"):
        fit_streaming(lambda: chunks(iris_quanti_df, 10), nf=0)


def test_fit_streaming_rejects_an_unknown_method(iris_quanti_df: pd.DataFrame) -> None:
    with pytest.raises(InvalidParameterException, match="method"):
        fit_streaming(lambda: chunks(iris_quanti_df, 10), nf=2, method="lda")


def test_fit_streaming_rejects_a_changing_schema(iris_quanti_df: pd.DataFrame) -> None:
    df = iris_quanti_df
    renamed = df.rename(columns={"sepal.length": "other"})

    def batches() -> Iterator[pd.DataFrame]:
        yield df.iloc[:50]
        yield renamed.iloc[50:]

    with pytest.raises(ValueError, match="same columns"):
        fit_streaming(batches, nf=2)


def test_fit_streaming_rejects_a_changing_dtype() -> None:
    def batches() -> Iterator[pd.DataFrame]:
        yield pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
        yield pd.DataFrame({"a": ["x", "y", "z"], "b": [4.0, 5.0, 6.0]})

    with pytest.raises(ValueError, match="changed dtype"):
        fit_streaming(batches, nf=1)


def test_fit_streaming_rejects_batches_that_differ_between_passes() -> None:
    """A factory that yields a different table on the second pass is not a fit."""
    df = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) ** 2})
    calls = []

    def batches() -> Iterator[pd.DataFrame]:
        calls.append(1)
        source = df if len(calls) == 1 else df.iloc[:10]
        yield from chunks(source, 4)

    with pytest.raises(ValueError, match="pass 1 counted"):
        fit_streaming(batches, nf=2)


def test_fit_streaming_rejects_datetime_columns() -> None:
    df = pd.DataFrame({"when": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])})

    with pytest.raises(ValueError, match="datetime"):
        fit_streaming(lambda: chunks(df, 2), nf=1)


def test_fit_streaming_rejects_zero_rows() -> None:
    with pytest.raises(ValueError, match="zero rows"):
        fit_streaming(lambda: iter([pd.DataFrame({"a": [], "b": []})]), nf=1)


def test_fit_streaming_skips_empty_batches(iris_quanti_df: pd.DataFrame) -> None:
    df = iris_quanti_df
    nf = min(df.shape)
    empty = df.iloc[:0]

    def batches() -> Iterator[pd.DataFrame]:
        yield empty
        yield from chunks(df, 40)
        yield empty

    reference = pca.fit(df, nf=nf)
    streamed = fit_streaming(batches, nf=nf)

    assert_agrees_with_reference(streamed, reference)


def test_accumulators_can_be_driven_directly(iris_quanti_df: pd.DataFrame) -> None:
    """The two-pass API is usable without the convenience wrapper."""
    df = iris_quanti_df
    nf = min(df.shape)

    scaling = ScalingAccumulator()
    for batch in chunks(df, 20):
        scaling.partial_fit(batch)
    params = scaling.finalize()

    assert params.method == "pca"
    assert params.p == 4
    assert params.n == len(df)

    decomposition = DecompositionAccumulator(params, nf=nf)
    for batch in chunks(df, 20):
        decomposition.partial_fit(batch)
    streamed = decomposition.finalize()

    assert_agrees_with_reference(streamed, pca.fit(df, nf=nf))


# ---------------------------------------------------------------------------
# FAMD
# ---------------------------------------------------------------------------


def streamed_rank(df: pd.DataFrame) -> int:
    """The rank the streaming fit reports for `df`, pinned against `fit` below."""
    return _scaling_params(df, size=64).max_rank


def whole_table_width(df: pd.DataFrame) -> int:
    """Number of columns of the scaled matrix.

    Not `min(pd.get_dummies(df).shape)`: that leaves a boolean column alone, while
    every `fit` casts it to a category first and gives it one dummy per value.
    """
    return _scaling_params(df, size=64).p


def reference_famd(df: pd.DataFrame, *, nf: int | None = None, **kwargs: object) -> Model:
    """Fit the whole table down the full-SVD path.

    `get_svd` takes the randomized path when `nf < 0.8 * min(shape)`, and that path
    is itself approximate, so a reference taken from it would not be one. Passing a
    smaller `nf` keeps the full path only because the reference is refitted here at
    full width and truncated afterwards.
    """
    width = min(len(df), whole_table_width(df))
    model = famd.fit(df, nf=width, **kwargs)  # type: ignore[arg-type]
    if nf is None:
        return model
    return _truncated(model, nf)


def _truncated(model: Model, nf: int) -> Model:
    """Keep `nf` components of a model fitted at full width."""
    assert model.s is not None
    model.V = model.V[:nf, :]
    model.s = model.s[:nf]
    model.explained_var = model.explained_var[:nf]
    model.explained_var_ratio = model.explained_var_ratio[:nf]
    model.variable_coord = pd.DataFrame(model.V.T)
    model.nf = nf
    return model


def mixed_table(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "num_1": rng.normal(size=n),
            "num_2": rng.normal(loc=50, scale=3, size=n),
            "tool": rng.choice(["wrench", "hammer", "saw"], size=n),
            "fruit": rng.choice(["apple", "orange"], size=n),
        }
    )


@pytest.mark.parametrize("size", [1, 3, 11, 97, 400, 1000])
def test_fit_streaming_famd_equals_fit(size: int) -> None:
    df = mixed_table()
    nf = streamed_rank(df)

    reference = reference_famd(df)
    streamed = fit_streaming(lambda: chunks(df, size), nf=nf)

    assert_agrees_with_reference(streamed, reference)


def test_fit_streaming_famd_independent_of_batch_order() -> None:
    df = mixed_table()
    nf = streamed_rank(df)
    shuffled = df.sample(frac=1, random_state=0)

    reference = reference_famd(df)
    streamed = fit_streaming(lambda: chunks(shuffled, 37), nf=nf)

    assert_agrees_with_reference(streamed, reference)


def test_fit_streaming_famd_with_col_weights() -> None:
    df = mixed_table()
    nf = streamed_rank(df)
    col_weights = np.array([3.0, 1.0, 2.0, 1.0])

    reference = reference_famd(df, col_weights=col_weights)
    streamed = fit_streaming(lambda: chunks(df, 29), nf=nf, col_weights=col_weights)

    assert_agrees_with_reference(streamed, reference)


def test_fit_streaming_famd_with_a_modality_only_in_the_last_batch() -> None:
    """A modality first seen in the last batch still gets its own column.

    This is what forces the two passes: the dummy column it needs did not exist
    while the earlier batches were scaled, and after centering its entries there
    are not zero.
    """
    df = mixed_table(n=120)
    df.loc[df.index[-1], "tool"] = "chisel"
    nf = streamed_rank(df)

    reference = reference_famd(df)
    streamed = fit_streaming(lambda: chunks(df, 20), nf=nf)

    assert f"tool{DUMMIES_SEPARATOR}chisel" in streamed.dummy_categorical
    assert_agrees_with_reference(streamed, reference)


def test_fit_streaming_famd_with_null_categorical_value() -> None:
    """A null takes no dummy column, in a batch exactly as in the whole table."""
    df = mixed_table(n=120)
    df.loc[df.index[5], "tool"] = None
    df.loc[df.index[63], "fruit"] = None
    nf = streamed_rank(df)

    reference = reference_famd(df)
    streamed = fit_streaming(lambda: chunks(df, 17), nf=nf)

    assert_agrees_with_reference(streamed, reference)


def test_fit_streaming_rejects_a_null_continuous_value() -> None:
    """No decomposition takes a null, and a whole-table fit rejects one too.

    `linalg.svd` refuses a matrix holding one, so the streamed fit says so at the
    batch that carries it rather than carrying it into the merged factor and
    failing at the end with a decomposition error.
    """
    df = mixed_table(n=120)
    df.loc[df.index[7], "num_1"] = np.nan

    with pytest.raises(ValueError, match="must not contain infs or NaNs"):
        reference_famd(df)

    with pytest.raises(ValueError, match="non-finite"):
        fit_streaming(lambda: chunks(df, 17), nf=4)


def test_fit_streaming_famd_with_constant_columns() -> None:
    df = mixed_table(n=120)
    df["num_constant"] = 4.0
    df["cat_constant"] = "only"
    nf = streamed_rank(df)

    reference = reference_famd(df)
    streamed = fit_streaming(lambda: chunks(df, 23), nf=nf)

    assert_agrees_with_reference(streamed, reference)


def test_fit_streaming_famd_on_a_boolean_column() -> None:
    """A boolean column is categorical, and its modalities are named True/False."""
    df = mixed_table(n=80)
    rng = np.random.default_rng(7)
    df["flag"] = rng.choice([True, False], size=len(df))
    nf = streamed_rank(df)

    reference = reference_famd(df)
    streamed = fit_streaming(lambda: chunks(df, 13), nf=nf)

    assert f"flag{DUMMIES_SEPARATOR}True" in streamed.dummy_categorical
    assert_agrees_with_reference(streamed, reference)


# ---------------------------------------------------------------------------
# The reported rank
# ---------------------------------------------------------------------------


def rank_cases() -> list[tuple[str, pd.DataFrame]]:
    df_null_categorical = mixed_table(n=120)
    df_null_categorical.loc[df_null_categorical.index[5], "tool"] = None

    df_boolean = mixed_table(n=80)
    df_boolean["flag"] = np.random.default_rng(7).choice([True, False], size=80)

    df_constant = mixed_table(n=120)
    df_constant["cat_constant"] = "only"

    return [
        ("mixed", mixed_table(n=120)),
        ("null categorical", df_null_categorical),
        ("boolean", df_boolean),
        ("constant categorical", df_constant),
    ]


@pytest.mark.parametrize(("name", "df"), rank_cases())
def test_reported_rank_matches_the_whole_table_fit(name: str, df: pd.DataFrame) -> None:
    """The rank `nf` is validated against must be the one `fit` actually finds.

    Too low and a legitimate `nf` is refused; too high and the fit returns axes
    that are an arbitrary basis of the null space.
    """
    reference = reference_famd(df)
    assert reference.s is not None

    assert streamed_rank(df) == numerical_rank(reference.s)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_famd_round_trip_matches_the_whole_table_round_trip() -> None:
    """transform then inverse_transform gives what the whole-table model gives.

    Stricter than comparing V: transform and inverse_transform use the same V, so
    a flipped axis cancels and only a real difference in the fit shows up.
    """
    df = mixed_table(n=120)
    nf = streamed_rank(df)

    reference = reference_famd(df, nf=nf)
    streamed = fit_streaming(lambda: chunks(df, 17), nf=nf)

    back_reference = inverse_transform(transform(df, reference), reference)
    back_streamed = inverse_transform(transform(df, streamed), streamed)

    assert_frame_equal(back_streamed, back_reference)


# ---------------------------------------------------------------------------
# MCA
# ---------------------------------------------------------------------------


def categorical_table(n: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tool": rng.choice(["wrench", "hammer", "saw"], size=n),
            "fruit": rng.choice(["apple", "orange"], size=n),
            "colour": rng.choice(["red", "green", "blue", "black"], size=n),
        }
    )


def reference_mca(df: pd.DataFrame, *, nf: int | None = None, **kwargs: object) -> Model:
    """Fit the whole table down the full-SVD path."""
    width = min(len(df), whole_table_width(df))
    model = mca.fit(df, nf=width, **kwargs)  # type: ignore[arg-type]
    if nf is None:
        return model
    assert model.D_c is not None
    D_c = model.D_c
    truncated = _truncated(model, nf)
    truncated.variable_coord = pd.DataFrame(D_c @ truncated.V.T)
    return truncated


def test_streamed_mca_scaling_equals_diag_compute() -> None:
    """The row-local formula must equal the code it replaces, not merely the SVD of it.

    `mca.center` and `mca._diag_compute` are what a whole-table fit decomposes, and
    they cannot stream: they build an `n x p` dense array and an `n x n` diagonal.
    """
    df = categorical_table(n=200)

    df_scale, _, r, c = mca.center(df)
    _, expected, _ = mca._diag_compute(df_scale, r, c)

    params = _scaling_params(df, size=64)
    decomposition = DecompositionAccumulator(params, nf=params.max_rank)
    streamed = np.vstack([decomposition._scale(batch) for batch in chunks(df, 7)])

    assert_allclose(streamed, np.asarray(expected), atol=1e-17)


@pytest.mark.parametrize("size", [1, 3, 11, 97, 400, 1000])
def test_fit_streaming_mca_equals_fit(size: int) -> None:
    df = categorical_table()
    nf = streamed_rank(df)

    reference = reference_mca(df)
    streamed = fit_streaming(lambda: chunks(df, size), nf=nf)

    assert_agrees_with_reference(streamed, reference)


def test_fit_streaming_mca_independent_of_batch_order() -> None:
    df = categorical_table()
    nf = streamed_rank(df)
    shuffled = df.sample(frac=1, random_state=0)

    reference = reference_mca(df)
    streamed = fit_streaming(lambda: chunks(shuffled, 31), nf=nf)

    assert_agrees_with_reference(streamed, reference)


def test_fit_streaming_mca_with_col_weights() -> None:
    df = categorical_table()
    nf = streamed_rank(df)
    col_weights = np.array([3.0, 1.0, 2.0])

    reference = reference_mca(df, col_weights=col_weights)
    streamed = fit_streaming(lambda: chunks(df, 23), nf=nf, col_weights=col_weights)

    assert_agrees_with_reference(streamed, reference)


def test_fit_streaming_mca_with_null_categorical_value() -> None:
    """The row share `r` is not constant once a column holds a null.

    A row with a null has fewer dummies set, so its share of the dummy matrix is
    smaller. Substituting `1/n` for it puts a 16% error on the singular values,
    which this equality would not survive.
    """
    df = categorical_table(n=200)
    df.loc[df.index[3], "tool"] = None
    df.loc[df.index[57], "colour"] = None
    df.loc[df.index[58], "fruit"] = None
    nf = streamed_rank(df)

    reference = reference_mca(df)
    streamed = fit_streaming(lambda: chunks(df, 13), nf=nf)

    assert_agrees_with_reference(streamed, reference)


def test_fit_streaming_mca_with_a_modality_only_in_the_last_batch() -> None:
    df = categorical_table(n=120)
    df.loc[df.index[-1], "colour"] = "violet"
    nf = streamed_rank(df)

    reference = reference_mca(df)
    streamed = fit_streaming(lambda: chunks(df, 20), nf=nf)

    assert f"colour{DUMMIES_SEPARATOR}violet" in streamed.dummy_categorical
    assert_agrees_with_reference(streamed, reference)


def test_fit_streaming_mca_with_a_constant_column() -> None:
    df = categorical_table(n=120)
    df["constant"] = "only"
    nf = streamed_rank(df)

    reference = reference_mca(df)
    streamed = fit_streaming(lambda: chunks(df, 17), nf=nf)

    assert_agrees_with_reference(streamed, reference)


def test_mca_round_trip_matches_the_whole_table_round_trip() -> None:
    df = categorical_table(n=120)
    nf = streamed_rank(df)

    reference = reference_mca(df, nf=nf)
    streamed = fit_streaming(lambda: chunks(df, 17), nf=nf)

    back_reference = inverse_transform(transform(df, reference), reference)
    back_streamed = inverse_transform(transform(df, streamed), streamed)

    assert_frame_equal(back_streamed, back_reference)
