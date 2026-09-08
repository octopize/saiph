"""Fit a projection from synthesised batches and report the peak resident size.

Run as a child process so the high-water mark belongs to this fit alone:

    python -m saiph.tests.peak_memory <row-count> <batch-size> <method>
"""

import resource
import sys
from collections.abc import Iterator

import numpy as np
import pandas as pd

from saiph.streaming import fit_streaming


def synthesise(n_rows: int, batch_size: int, method: str) -> Iterator[pd.DataFrame]:
    """Yield batches without ever building the whole table."""
    rng = np.random.default_rng(0)
    produced = 0
    while produced < n_rows:
        size = min(batch_size, n_rows - produced)
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


def peak_resident_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kibibytes on Linux.
    return usage if sys.platform == "darwin" else usage * 1024


def main() -> None:
    n_rows, batch_size, method = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    fit_streaming(lambda: synthesise(n_rows, batch_size, method), nf=2, method=method)
    print(peak_resident_bytes())  # noqa: T201


if __name__ == "__main__":
    main()
