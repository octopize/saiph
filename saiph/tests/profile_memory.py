import time
import tracemalloc
import webbrowser
import time
from pathlib import Path

import numpy as np
import pandas as pd
# from scalene import scalene_profiler
from filprofiler.api import profile

from saiph import fit
from saiph.lib.size import get_readable_size

N_ROWS = 10000

# Run with:
# poetry run fil-profile python saiph/tests/profile_memory.py

BASE_PATH = (Path(__file__).parent / "../../").resolve()


def main() -> None:
    quali = np.random.randint(0, 1000, size=N_ROWS).astype(object)
    quanti = np.random.randint(0, 1000, size=N_ROWS, dtype=np.int32)

    df = pd.DataFrame({"quali": quali, "quanti": quanti})
    print(f"using {get_readable_size(df.memory_usage(index=True).sum())}")

    # tracemalloc.start()
    # scalene_profiler.start()
    print("before fit")
    start = time.perf_counter()
    # fit(df, nf=5)
    filename = f"fil-result/{time.time()}"
    full_path = BASE_PATH / filename / 'index.html'
    print(full_path)
    profile(lambda: fit(df, nf=5), filename)
    end = time.perf_counter()

    print(f"after fit, took {(end-start):.3} sec")

    webbrowser.open(f"file://{str(full_path)}")
    # scalene_profiler.stop()

    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')
    # [print(t) for t in top_stats[:30]]

    # _, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # print(get_readable_size(peak))

    # memory usage should be below x kiB
    # assert peak < 300 * 1024
    # record_property("peak_memory_usage", peak)


if __name__ == "__main__":
    main()
