import tracemalloc
from pathlib import Path
from typing import Any

import pandas as pd

from saiph.projection import fit


def test_memory_iris(record_property: Any) -> None:
    path = (Path(__file__) / "../../../fixtures/iris.csv").resolve()
    iris_df = pd.read_csv(path)

    tracemalloc.start()
    fit(iris_df)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # memory usage should be below x kiB
    assert peak <= 64276
    record_property("peak_memory_usage", peak)
