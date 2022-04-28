from pathlib import Path

import numpy as np
import pandas as pd

from saiph import fit

# flake8: noqa

N_ROWS = 1000000

# Run with:
# poetry run py-spy record -f speedscope -o "saiph/tmp/profile_$(date)" -- python saiph/tests/profile_cpu.py

BASE_PATH = (Path(__file__).parent / "../").resolve()


def main() -> None:
    print(BASE_PATH)
    df = pd.read_csv(str(BASE_PATH) + "/tmp/fake_1000000.csv")

    fit(df, nf=5)

    fit(df, nf=5, sparse=True)


if __name__ == "__main__":
    main()
