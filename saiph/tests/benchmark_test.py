import tracemalloc
from collections.abc import Iterator
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from typing import Any

import numpy as np
import pandas as pd
import pytest

from saiph.projection import fit
from saiph.streaming import fit_streaming

# 2024-07-03 @chybz
# - waaayyy to tied to Python version and other modules
#   --> IMO, should disappear...
# - making this obvious and adjustable
MAX_MEM_BYTES: int = 200000


def test_memory_iris(record_property: Any, iris_df: pd.DataFrame) -> None:
    fit(iris_df)
    peak = int(getrusage(RUSAGE_SELF).ru_maxrss / 1024)
    # memory usage should be below x kiB
    error_margin = 1.1
    assert peak <= MAX_MEM_BYTES * error_margin
    record_property("peak_memory_usage", peak)


def test_memory_iris_sparse(record_property: Any, iris_df: pd.DataFrame) -> None:
    fit(iris_df, sparse=True)
    peak = int(getrusage(RUSAGE_SELF).ru_maxrss / 1024)
    # memory usage should be below x kiB
    error_margin = 1.1
    assert peak <= MAX_MEM_BYTES * error_margin
    record_property("peak_memory_usage", peak)


def test_1k(benchmark: Any) -> None:
    # This file does not exist on CI
    path = (Path(__file__) / "../../../tmp/fake_1k.csv").resolve()
    df = pd.read_csv(path)

    benchmark(fit, df)


def test_10k(benchmark: Any) -> None:
    # This file does not exist on CI
    path = (Path(__file__) / "../../../tmp/fake_10k.csv").resolve()
    df = pd.read_csv(path)

    benchmark(fit, df)


@pytest.mark.slow_benchmark
def test_1m(benchmark: Any) -> None:
    # This file does not exist on CI
    path = (Path(__file__) / "../../../tmp/fake_1000000.csv").resolve()
    df = pd.read_csv(path)

    benchmark(fit, df)


# Sixteen times the rows. tracemalloc counts bytes exactly, so the two points need only
# be far enough apart that holding the table would show; they do not need to be large.
SMALL_ROW_COUNT = 2_000
LARGE_ROW_COUNT = 32_000
STREAMING_BATCH_SIZE = 1_000
# Columns of the scaled matrix per method.
SCALED_WIDTH = {"pca": 2, "famd": 7, "mca": 5}


def synthesise(n_rows: int, method: str) -> Iterator[pd.DataFrame]:
    rng = np.random.default_rng(0)
    produced = 0
    while produced < n_rows:
        size = min(STREAMING_BATCH_SIZE, n_rows - produced)
        produced += size
        continuous = {
            "num_1": rng.normal(size=size),
            "num_2": rng.normal(loc=50, scale=3, size=size),
        }
        categorical = {
            "tool": rng.choice(["wrench", "hammer", "saw"], size=size),
            "fruit": rng.choice(["apple", "orange"], size=size),
        }
        if method == "pca":
            yield pd.DataFrame(continuous)
        elif method == "mca":
            yield pd.DataFrame(categorical)
        else:
            yield pd.DataFrame({**continuous, **categorical})


def measure_peak_bytes(n_rows: int, method: str) -> int:
    """Bytes allocated at peak by one streaming fit.

    tracemalloc rather than ru_maxrss, which is a whole-process high-water mark: it
    would carry whatever an earlier test allocated, and varies by ~13 MB between runs.
    """
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        fit_streaming(lambda: synthesise(n_rows, method), nf=2, method=method)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


@pytest.mark.parametrize("method", ["pca", "famd", "mca"])
def test_fit_streaming_peak_memory_is_flat_in_row_count(record_property: Any, method: str) -> None:
    """Sixteen times the rows must not cost sixteen times the memory.

    All the growth there is is Model.row_weights, one float per individual, so the
    bound is that vector. A fit holding the table would exceed it on the scaled matrix
    alone, before the dataframe.
    """
    growth = measure_peak_bytes(LARGE_ROW_COUNT, method) - measure_peak_bytes(
        SMALL_ROW_COUNT, method
    )
    row_weights_growth = (LARGE_ROW_COUNT - SMALL_ROW_COUNT) * 8

    record_property(f"peak_memory_growth_{method}", growth)
    assert growth < 1.5 * row_weights_growth
    assert growth < LARGE_ROW_COUNT * SCALED_WIDTH[method] * 8
