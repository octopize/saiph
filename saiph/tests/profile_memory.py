# type: ignore
import time
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from filprofiler.api import profile

from saiph import fit
from saiph.lib.size import get_readable_size

# flake8: noqa

N_ROWS = 10000

# Run with:
# poetry run fil-profile python saiph/tests/profile_memory.py

BASE_PATH = (Path(__file__).parent / "../").resolve()


def main() -> None:
    df = pd.read_csv(str(BASE_PATH) + "/tmp/fake_1000000.csv")

    print(f"using {get_readable_size(df.memory_usage(index=True).sum())}")

    print("before fit")
    start = time.perf_counter()
    # fit(df, nf=5)
    filename = f"/tmp/{time.time()}"
    full_path = BASE_PATH / filename / "index.html"
    print(full_path)
    profile(lambda: fit(df, nf=5), filename)
    end = time.perf_counter()

    print(f"after fit, took {(end-start):.3} sec")

    webbrowser.open(f"file://{str(full_path)}")

    print("before fit 2")
    start = time.perf_counter()
    # fit(df, nf=5)
    filename = f"/tmp/{time.time()}"
    full_path = BASE_PATH / filename / "index.html"
    print(full_path)
    profile(lambda: fit(df, nf=5, sparse=True), filename)
    end = time.perf_counter()

    print(f"after fit, took {(end-start):.3} sec")

    webbrowser.open(f"file://{str(full_path)}")


if __name__ == "__main__":
    main()
