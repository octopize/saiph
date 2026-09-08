"""Fit a projection from batches of rows, without ever holding the whole table.

Two passes. Pass 1 accumulates the scaling constants that every row of pass 2
needs — the mean, the standard deviation, the modality counts. Pass 2 scales
each batch and folds it into a carried QR factor `R`.

`R` is what makes the result exact rather than approximate. Writing the scaled
matrix as `Z = Q R` with orthonormal `Q` gives `Zᵀ Z = Rᵀ R`, and the singular
values and right singular vectors are determined by `Zᵀ Z` alone, so `R` carries
them without loss. Stacking the next batch under the carried `R` and refactoring
keeps that true for every row seen, so the decomposition at the end is the one
`fit` would have computed on the whole table.

The left singular vectors are the one part of the decomposition with a row per
individual. They are not recoverable from `R`, and this is deliberate: a matrix
of that size is what makes a whole-table fit impossible in the first place.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.utils import extmath

from saiph.exception import InvalidParameterException
from saiph.models import Model
from saiph.reduction import DUMMIES_SEPARATOR, famd, pca
from saiph.reduction.utils.common import (
    expand_column_weights,
    get_explained_variance,
    get_modalities_types,
    get_uniform_row_weights,
)

METHODS = ("pca", "famd", "mca")

_EPS: np.float64 = np.finfo(float).eps


@dataclass
class ScalingParams:
    """Everything pass 2 needs to scale a batch, and pass 1 is the only way to know it."""

    method: str
    n: int
    original_dtypes: pd.Series
    quanti: list[str]
    quali: list[str]
    modalities_types: dict[str, str]
    mean: pd.Series
    std: pd.Series
    modalities: NDArray[Any]
    column_weights: NDArray[np.float64]
    # Number of individuals taking each modality, indexed and ordered as `modalities`.
    modality_counts: pd.Series

    @property
    def p(self) -> int:
        """Number of columns of the scaled matrix."""
        return len(self.quanti) + len(self.modalities)

    @property
    def total(self) -> float:
        """Number of ones in the whole dummy matrix, `X.sum().sum()`."""
        return float(self.modality_counts.sum())

    @property
    def column_masses(self) -> NDArray[np.float64]:
        """Share of the dummy matrix held by each modality, `c` in the MCA scaling."""
        return np.asarray(self.modality_counts / self.total, dtype=np.float64)

    @property
    def D_c(self) -> NDArray[np.float64]:
        return np.diag(1 / (_EPS + np.sqrt(self.column_masses)))

    @property
    def dummies_col_prop(self) -> NDArray[np.float64]:
        """Number of individuals per individual taking each modality."""
        return np.asarray(self.n / self.modality_counts, dtype=np.float64)

    @property
    def prop(self) -> pd.Series:
        """Proportion of individuals taking each modality.

        A null takes no dummy column, so a column holding one has modality
        proportions summing to less than 1.
        """
        return self.modality_counts / self.n

    @property
    def max_rank(self) -> int:
        """Upper bound on the rank of the scaled matrix.

        The dummies of a categorical variable sum to one on every row, so after
        centering they are linearly dependent and the variable costs one direction:
        the axes past this count are an arbitrary basis of the null space, not a
        decomposition of the data.

        A null breaks that dependency, because it takes no dummy column and so
        leaves a row whose dummies sum to zero. A variable holding one therefore
        keeps its full set of directions.
        """
        complete = sum(1 for col in self.quali if self._is_complete(col))
        return min(self.n - 1, self.p - complete)

    def _is_complete(self, col: str) -> bool:
        """Whether every individual has a value for `col`."""
        prefix = f"{col}{DUMMIES_SEPARATOR}"
        counts = self.modality_counts[
            [name for name in self.modality_counts.index if name.startswith(prefix)]
        ]
        return bool(counts.sum() == self.n)

    def to_model(self) -> Model:
        """Build the model with the fields pass 1 determines.

        The decomposition fields are left empty for `DecompositionAccumulator.finalize`
        to fill in. Pass 2 scales its batches through this same object, so it goes
        down the same `scaler` the fitted model will use at transform time.
        """
        model = Model(
            original_dtypes=self.original_dtypes,
            original_categorical=self.quali,
            original_continuous=self.quanti,
            dummy_categorical=list(self.modalities),
            modalities_types=self.modalities_types,
            mean=self.mean if self.quanti else None,
            std=self.std if self.quanti else None,
            _modalities=self.modalities if len(self.modalities) else None,
            column_weights=self.column_weights,
            type=self.method,
            U=np.empty((0, 0)),
            V=np.empty((0, 0)),
            explained_var=np.empty(0),
            explained_var_ratio=np.empty(0),
            variable_coord=pd.DataFrame(),
            row_weights=np.empty(0),
            nf=0,
        )
        if self.method == "famd":
            model.prop = self.prop
        if self.method == "mca":
            model.D_c = self.D_c
            model.dummies_col_prop = self.dummies_col_prop
        return model


class ScalingAccumulator:
    """Pass 1: accumulate the scaling constants over batches of rows.

    The schema and the choice of method are frozen on the first batch. A later
    batch that disagrees is an error rather than a silent change of model.
    """

    def __init__(
        self,
        *,
        method: str | None = None,
        col_weights: NDArray[np.float64] | None = None,
    ) -> None:
        if method is not None and method not in METHODS:
            raise InvalidParameterException(
                f"Expected 'method' to be one of {METHODS} or None, got {method!r} instead."
            )
        self._requested_method = method
        self._col_weights = col_weights

        self._n = 0
        self._columns: list[str] | None = None
        self._original_dtypes: pd.Series | None = None
        self._quanti: list[str] = []
        self._quali: list[str] = []
        self._modalities_types: dict[str, str] = {}
        self._mean: NDArray[np.float64] = np.zeros(0)
        self._var: NDArray[np.float64] = np.zeros(0)
        self._seen: NDArray[np.float64] = np.zeros(0)
        self._counts: dict[str, pd.Series] = {}

    def partial_fit(self, batch: pd.DataFrame) -> None:
        """Accumulate one batch of rows."""
        if self._columns is None:
            self._freeze_schema(batch)
        else:
            self._check_schema(batch)

        if len(batch) == 0:
            return

        self._n += len(batch)

        if self._quanti:
            values = batch[self._quanti].to_numpy(dtype=np.float64)
            self._mean, self._var, self._seen = extmath._incremental_mean_and_var(
                values, self._mean, self._var, self._seen
            )

        for col in self._quali:
            # value_counts drops nulls, as pd.get_dummies does.
            counts = batch[col].value_counts()
            self._counts[col] = self._counts[col].add(counts, fill_value=0)

    def finalize(self) -> ScalingParams:
        """Close pass 1 and return the constants pass 2 needs."""
        if self._columns is None or self._original_dtypes is None:
            raise ValueError("No batch was accumulated. Call partial_fit() at least once.")
        if self._n == 0:
            raise ValueError("Cannot fit on zero rows.")

        modality_counts = self._ordered_modality_counts()
        modalities = np.array(modality_counts.index.to_list(), dtype=object)

        col_weights = (
            np.ones(len(self._columns)) if self._col_weights is None else self._col_weights
        )
        column_weights = expand_column_weights(
            col_weights,
            self._columns,
            self._quanti,
            self._quali,
            list(modalities),
        )

        return ScalingParams(
            method=self._method(),
            n=self._n,
            original_dtypes=self._original_dtypes,
            quanti=self._quanti,
            quali=self._quali,
            modalities_types=self._modalities_types,
            mean=pd.Series(self._mean, index=self._quanti),
            std=pd.Series(np.sqrt(self._var), index=self._quanti),
            modalities=modalities,
            column_weights=column_weights,
            modality_counts=modality_counts,
        )

    def _freeze_schema(self, batch: pd.DataFrame) -> None:
        self._columns = batch.columns.to_list()
        self._original_dtypes = batch.dtypes
        self._quanti = batch.select_dtypes(include=["int", "float", "number"]).columns.to_list()
        self._quali = batch.select_dtypes(exclude=["int", "float", "number"]).columns.to_list()

        datetime_cols = batch.select_dtypes(include=["datetime", "datetimetz"]).columns.to_list()
        if datetime_cols:
            raise ValueError(
                f"DataFrame contains datetime column(s): {datetime_cols}. "
                "Convert them to numeric (e.g. seconds since epoch) before fitting."
            )

        self._mean = np.zeros(len(self._quanti))
        self._var = np.zeros(len(self._quanti))
        self._seen = np.zeros(len(self._quanti))
        self._counts = {col: pd.Series(dtype=np.float64) for col in self._quali}

        if self._quali and len(batch) > 0:
            # get_modalities_types reads the row labelled 0, which only the first
            # batch of an arbitrarily indexed source is guaranteed to have.
            quali_batch = batch[self._quali].reset_index(drop=True)
            self._modalities_types = get_modalities_types(quali_batch)

        if self._col_weights is not None and len(self._col_weights) != len(self._columns):
            raise InvalidParameterException(
                f"Expected one column weight per column, got {len(self._col_weights)} "
                f"weights for {len(self._columns)} columns."
            )

    def _check_schema(self, batch: pd.DataFrame) -> None:
        if batch.columns.to_list() != self._columns:
            raise ValueError(
                "Expected every batch to have the same columns in the same order. "
                f"Got {batch.columns.to_list()}, expected {self._columns}."
            )
        differing = [
            col
            for col, dtype in batch.dtypes.items()
            if dtype != self._original_dtypes[col]  # type: ignore[index]
        ]
        if differing:
            raise ValueError(
                f"Expected every batch to have the same dtypes. Column(s) {differing} "
                "changed dtype between batches, which would change the fitted model."
            )

    def _method(self) -> str:
        if self._requested_method is not None:
            return self._requested_method
        if not self._quali:
            return "pca"
        if not self._quanti:
            return "mca"
        return "famd"

    def _ordered_modality_counts(self) -> pd.Series:
        """Counts per dummy column, named and ordered as pd.get_dummies would."""
        per_column = []
        for col in self._quali:
            counts = self._counts[col]
            categories = pd.Series(counts.index.to_list()).astype("category").cat.categories
            counts = counts.reindex(categories)
            # Take the names from get_dummies itself rather than formatting them here,
            # so a non-string modality is named the same way in both paths.
            names = pd.get_dummies(
                pd.Series(pd.Categorical(categories, categories=categories), name=col).to_frame(),
                prefix_sep=DUMMIES_SEPARATOR,
                dtype=np.uint8,
            ).columns.to_list()
            per_column.append(pd.Series(counts.to_numpy(), index=names, dtype=np.float64))

        if not per_column:
            return pd.Series(dtype=np.float64)
        return pd.concat(per_column)


class DecompositionAccumulator:
    """Pass 2: scale each batch and fold it into the carried QR factor."""

    def __init__(
        self,
        params: ScalingParams,
        nf: int,
        *,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if nf <= 0 or nf > params.max_rank:
            raise InvalidParameterException(
                "Expected number of components to be in "
                f"0 < 'nf' <= {params.max_rank}, got {nf} instead."
            )
        self.params = params
        self.nf = nf
        self._random_gen = (
            seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        )
        self._model = params.to_model()
        self._R: NDArray[np.float64] | None = None
        self._rows_seen = 0

    def partial_fit(self, batch: pd.DataFrame) -> None:
        """Scale one batch of rows and merge it into the carried factor."""
        if len(batch) == 0:
            return
        self._rows_seen += len(batch)
        if self._rows_seen > self.params.n:
            raise ValueError(
                f"Pass 2 has seen more rows ({self._rows_seen}) than pass 1 counted "
                f"({self.params.n}). The batches must be the same on both passes."
            )

        scaled = self._scale(batch)
        if not np.isfinite(scaled).all():
            raise ValueError(
                "The scaled batch holds non-finite values, which no decomposition "
                "accepts. A null or an infinity in a continuous column is rejected "
                "by a whole-table fit for the same reason."
            )
        weighted = scaled * self.params.column_weights / self.params.n

        stacked = weighted if self._R is None else np.vstack([self._R, weighted])
        self._R = cast(NDArray[np.float64], np.linalg.qr(stacked, mode="r"))

    def finalize(self) -> Model:
        """Decompose the carried factor and return the fitted model."""
        if self._R is None:
            raise ValueError("No batch was accumulated. Call partial_fit() at least once.")
        if self._rows_seen != self.params.n:
            raise ValueError(
                f"Pass 2 saw {self._rows_seen} rows but pass 1 counted {self.params.n}. "
                "The batches must be the same on both passes."
            )

        _, S, Vt = np.linalg.svd(self._R, full_matrices=False)
        # The U-based decision sklearn defaults to needs the left singular vectors,
        # which a streaming fit does not have.
        _, Vt = extmath.svd_flip(None, Vt, u_based_decision=False)
        if self.params.method != "mca":
            # mca.fit leaves the column weights in its right singular vectors, and
            # mca.transform carries D_c to compensate. Dividing them out here would
            # rescale every axis by 1/sqrt(weight).
            Vt = Vt / np.sqrt(self.params.column_weights)

        # S holds every singular value, so the ratio is against the true total
        # variance rather than against the truncated sum a partial SVD would give.
        explained_var, explained_var_ratio = get_explained_variance(S, self.params.n, self.nf)

        model = self._model
        model.V = Vt[: self.nf, :]
        model.s = S[: self.nf]
        model.explained_var = explained_var
        model.explained_var_ratio = explained_var_ratio
        if self.params.method == "mca":
            model.variable_coord = pd.DataFrame(self.params.D_c @ model.V.T)
        else:
            model.variable_coord = pd.DataFrame(model.V.T)
        model.row_weights = get_uniform_row_weights(self.params.n)
        model.nf = self.nf
        model.seed = int(self._random_gen.integers(0, 2**32 - 1))
        model.is_fitted = True
        return model

    def _scale(self, batch: pd.DataFrame) -> NDArray[np.float64]:
        """Scale a batch exactly as `fit` scales the whole table.

        Every constant this needs comes from pass 1, which is what makes the
        operation row-local and so batchable.
        """
        if self.params.method == "pca":
            scaled = pca.scaler(self._model, batch)
            return np.asarray(scaled, dtype=np.float64)

        if self.params.method == "famd":
            # The same scaler transform() calls, so a fit and a transform of the
            # same rows cannot drift apart.
            scaled = famd.scaler(self._model, batch)
            return np.asarray(scaled, dtype=np.float64)

        if self.params.method == "mca":
            return self._scale_mca(batch)

        raise NotImplementedError(f"Unsupported method {self.params.method!r}.")

    def _scale_mca(self, batch: pd.DataFrame) -> NDArray[np.float64]:
        """Scale a batch of dummies the way `mca.center` and `mca._diag_compute` do.

        Those two build an `n x p` dense array and an `n x n` diagonal, so MCA is the
        method that runs out of memory first. Written per row instead:

            T_ij = (X_ij / total - r_i c_j)
                   / ((eps + sqrt(r_i)) (eps + sqrt(c_j)))

        with `total` the number of ones in the whole dummy matrix, `c` the share of
        it held by each modality, and `r_i` this row's share of it. `total` and `c`
        come from pass 1; `r_i` is this row's own dummies, so nothing here reaches
        outside the row.

        `r_i` must stay per row. It equals `1/n` only while every individual has a
        value in every column: pd.get_dummies emits no indicator for a null, so a
        row holding one sums to less. Substituting `1/n` puts a 16% error on the
        singular values of a table with nulls.
        """
        dummies = pd.get_dummies(
            batch.astype("category"),
            prefix_sep=DUMMIES_SEPARATOR,
            dtype=np.uint8,
        )
        X = dummies.reindex(columns=self.params.modalities, fill_value=np.uint8(0)).to_numpy(
            dtype=np.float64
        )
        total = self.params.total
        c = self.params.column_masses

        r = X.sum(axis=1) / total
        centered = X / total - np.outer(r, c)
        scaled = centered / (_EPS + np.sqrt(c))
        row_scaled: NDArray[np.float64] = scaled / (_EPS + np.sqrt(r))[:, np.newaxis]
        return row_scaled


def fit_streaming(
    batches: Callable[[], Iterable[pd.DataFrame]],
    nf: int,
    *,
    col_weights: NDArray[np.float64] | None = None,
    method: str | None = None,
    seed: int | np.random.Generator | None = None,
) -> Model:
    """Fit a PCA, MCA or FAMD model from batches of rows.

    Datetimes must be stored as numbers of seconds since epoch.

    Parameters:
        batches: A callable returning a fresh iterable of batches. It is called
            twice, once per pass, so a bare iterator will not do.
        nf: Number of components to keep.
        col_weights: Weight assigned to each variable in the projection
            (more weight = more importance in the axes). One per original column.
        method: "pca", "mca" or "famd". Inferred from the dtypes of the first
            batch when not given.
        seed: Seed stored on the model for later use by inverse_transform.

    Returns:
        model: The model for transforming new data.
    """
    if not callable(batches):
        raise InvalidParameterException(
            "Expected 'batches' to be a callable returning a fresh iterable of "
            f"batches, got {type(batches).__name__} instead. An iterator would be "
            "exhausted by the first of the two passes, and the fit would silently "
            "see no rows on the second."
        )

    scaling = ScalingAccumulator(method=method, col_weights=col_weights)
    for batch in batches():
        scaling.partial_fit(batch)
    params = scaling.finalize()

    decomposition = DecompositionAccumulator(params, nf=nf, seed=seed)
    for batch in batches():
        decomposition.partial_fit(batch)
    return decomposition.finalize()
