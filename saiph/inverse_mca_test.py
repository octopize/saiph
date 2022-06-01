from saiph.inverse_transform import inverse_transform
from saiph.projection import fit
from pathlib import Path
import pandas as pd
from pandas.testing import assert_series_equal
import typer
from datetime import datetime
from saiph.conftest import _wbcd_csv, _wbcd_supplemental_coordinates_csv_mca
from saiph.test_utils import get_filenames, to_csv, set_active_user
import subprocess
import platform

def main(name : str):

    set_active_user(name)
    df = _wbcd_csv.drop(columns=["Sample_code_number"]).astype("category").copy()
    coordinates = _wbcd_supplemental_coordinates_csv_mca.copy()
    
    model = fit(df, nf="all")

    reversed_individuals = inverse_transform(coordinates, model)
    
    reversed_individuals = reversed_individuals.astype("int")

    to_csv(reversed_individuals, "reversed_individuals")

    now = datetime.now().strftime("%m-%d_%Hh%M")

    archive_name = f"debug_result_{name}_{now}.tar.gz"
    print(platform.system())
    tar_command = "tar" if platform.system() == "Linux" else "gtar"

    if tar_command != "tar":
        print("This program use GNU tar. If you are on Mac, you can install it using 'brew install gnu-tar'.")
        input("Press Enter to continue if you have installed it...")

    process = subprocess.run([tar_command, "czfv", archive_name, *get_filenames(), "--remove-files"],capture_output = True) 

    if process.returncode == 0:
        print(f"Success! Created archive '{archive_name}'")
        return 0
    print("FAILED!")
    return 1
if __name__ == "__main__":
    typer.run(main)

