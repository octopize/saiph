from pathlib import Path

import pandas as pd

from saiph import fit

N_ROWS = 1000000

# Run with:
# poetry run py-spy record -f speedscope -o "saiph/tmp/profile_$(date)"
# -- python saiph/tests/profile_cpu.py

BASE_PATH = (Path(__file__).parent / "../").resolve()


def main() -> None:
    """Run famd and sparse famd on a fake dataset."""
    df = pd.read_csv(str(BASE_PATH) + "/tmp/fake_1000000.csv")

    fit(df, nf=5)

    fit(df, nf=5, sparse=True)


if __name__ == "__main__":
    main()
