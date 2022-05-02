import tracemalloc
from pathlib import Path
from typing import Any

import pandas as pd

from saiph.projection import fit

from resource import getrusage, RUSAGE_SELF


def test_memory_iris(record_property: Any) -> None:
    path = (Path(__file__) / "../../../fixtures/iris.csv").resolve()
    iris_df = pd.read_csv(path)
  
    fit(iris_df)
    peak = int(getrusage(RUSAGE_SELF).ru_maxrss / 1024)
    # memory usage should be below x kiB
    assert peak <= 109872 * 1.01
    record_property("peak_memory_usage", peak)
