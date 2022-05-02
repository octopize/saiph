from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from typing import Any

import pandas as pd
import pytest

from saiph.projection import fit


def test_memory_iris(record_property: Any) -> None:
    path = (Path(__file__) / "../../../fixtures/iris.csv").resolve()
    iris_df = pd.read_csv(path)

    fit(iris_df)
    peak = int(getrusage(RUSAGE_SELF).ru_maxrss / 1024)
    # memory usage should be below x kiB
    assert peak <= 119200 * 1.05
    record_property("peak_memory_usage", peak)


def test_memory_iris_sparse(record_property: Any) -> None:
    path = (Path(__file__) / "../../../fixtures/iris.csv").resolve()
    iris_df = pd.read_csv(path)

    fit(iris_df, sparse=True)
    peak = int(getrusage(RUSAGE_SELF).ru_maxrss / 1024)
    # memory usage should be below x kiB
    assert peak <= 114400 * 1.05
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