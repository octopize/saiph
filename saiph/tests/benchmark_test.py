import subprocess
import sys
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from typing import Any

import pandas as pd
import pytest

from saiph.projection import fit

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


# ---------------------------------------------------------------------------
# Streaming fit: peak memory against row count
# ---------------------------------------------------------------------------

SMALL_ROW_COUNT = 50_000
LARGE_ROW_COUNT = 800_000
STREAMING_BATCH_SIZE = 10_000
# Columns of the scaled matrix per method, to size what a whole-table fit would hold.
SCALED_WIDTH = {"pca": 2, "famd": 7, "mca": 5}


def measure_peak_bytes(n_rows: int, method: str) -> int:
    """Peak resident size of a child process that fits `n_rows` and nothing else.

    A child process is what makes this measurable: ru_maxrss is a high-water mark
    for the whole process, so measuring in the test runner would report whichever
    earlier test allocated most.
    """
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "saiph.tests.peak_memory",
            str(n_rows),
            str(STREAMING_BATCH_SIZE),
            method,
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return int(completed.stdout.strip())


@pytest.mark.parametrize("method", ["pca", "famd", "mca"])
def test_fit_streaming_peak_memory_is_flat_in_row_count(record_property: Any, method: str) -> None:
    """Sixteen times the rows must not cost sixteen times the memory.

    This is the point of the feature, so it is asserted rather than assumed. The
    bound is the size of the scaled matrix a whole-table fit would have to hold,
    which is the smallest of the arrays that a streaming fit does not allocate.

    The remaining growth is Model.row_weights, one float per individual.
    """
    small = measure_peak_bytes(SMALL_ROW_COUNT, method)
    large = measure_peak_bytes(LARGE_ROW_COUNT, method)

    growth = large - small
    whole_table_scaled_matrix = LARGE_ROW_COUNT * SCALED_WIDTH[method] * 8
    row_weights = LARGE_ROW_COUNT * 8

    record_property(f"peak_memory_growth_{method}", growth)
    assert growth < whole_table_scaled_matrix
    assert growth < 4 * row_weights
