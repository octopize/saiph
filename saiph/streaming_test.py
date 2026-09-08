from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from pandas.testing import assert_series_equal

from saiph.exception import InvalidParameterException
from saiph.models import Model
from saiph.reduction import pca
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
