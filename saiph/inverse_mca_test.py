import platform
from cProfile import label
from math import ceil, sqrt
import re
from typing import List, Tuple
from saiph.inverse_transform import inverse_transform
from saiph.projection import fit
from pathlib import Path
import pandas as pd
from pandas.testing import assert_series_equal
import typer
from datetime import datetime
from saiph.conftest import _wbcd_csv, _wbcd_supplemental_coordinates_csv_mca
from saiph.test_utils import get_filenames, set_debug_mode, to_csv, set_active_user
import subprocess
import platform
from matplotlib import pyplot as plt

import tempfile 

from thefuzz import process

app = typer.Typer()

@app.command("generate")
def generate_debug_files(name : str, compress : bool = True):

    set_active_user(name)
    set_debug_mode(True)

    df = _wbcd_csv.drop(columns=["Sample_code_number"]).astype("category").copy()
    coordinates = _wbcd_supplemental_coordinates_csv_mca.copy()
    model = fit(df, nf="all")
    reversed_individuals = inverse_transform(coordinates, model)
    reversed_individuals = reversed_individuals.astype("int")

    to_csv(reversed_individuals, "reversed_individuals")

    if compress:
        return generate_archive(name)
    
    return 0

@app.command("explore")
def build_histogram():
    set_debug_mode(False)

    df = _wbcd_csv.drop(columns=["Sample_code_number"]).astype("category").copy()
    coordinates = _wbcd_supplemental_coordinates_csv_mca.copy()
    model = fit(df, nf="all")
    reversed_df = inverse_transform(coordinates, model)

    reversed_df = reversed_df.astype("int")
    df = df.astype("int")

    grid_size = ceil(sqrt(len(df.columns)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(20, 20))

    for ax, col in zip(axs.flat, df.columns):
        print(ax)
        df.hist(col, ax=ax , alpha=0.5, label="Original")
        reversed_df.hist(col, ax=ax,alpha=0.5, label = "Inverse")
        ax.legend()
    fig.suptitle(f"Inverse transform on {platform.system()}", fontsize=20)

    fig.savefig(f"distributions_{platform.system()}.svg")

    plt.show()


def generate_archive(name : str):
    """Generate an archive of the files and delete them after. """
    now = datetime.now().strftime("%m-%d_%Hh%M")

    archive_name = f"debug_result_{name}_{now}.tar.gz"
    tar_command = "tar" if platform.system() == "Linux" else "gtar"

    if tar_command != "tar":
        print("This program uses GNU tar. If you are on Mac, you can install it using 'brew install gnu-tar'.")
        input("Press Enter to continue if you have installed it...")

    process = subprocess.run([tar_command, "czf", archive_name, *get_filenames(), "--remove-files"]) 

    if process.returncode == 0:
        print(f"Success! Created archive '{archive_name}'")
        return 0
    print("FAILED!")

    return 1

def extract_archive(archive : Path, to : Path, tar_command : str):
    process = subprocess.run([tar_command, "xzf", archive, f"--directory={to}"]) 
    if process.returncode != 0:
        raise Exception("There was an ERROR. Exiting ...")
    
    print(f"Extracted archive '{archive.name}' to {to}")

def remove_date(filename : str) -> str:
    s = re.sub("\d", repl="", string=filename)
    s = re.sub("-_:", repl="", string=s)
    return s

@app.command("compare")
def compare(path_to_archive_1 : Path, path_to_archive_2 : Path):

    print("This program uses 'meld' to compare files. Make sure you have it installed.\n")
    input("Press Enter to continue if you have installed it...")

    tar_command = "tar" if platform.system() == "Linux" else "gtar"

    if tar_command != "tar":
        print("This program uses GNU tar. If you are on Mac, you can install it using 'brew install gnu-tar'.")
        input("Press Enter to continue if you have installed it...")


    with tempfile.TemporaryDirectory() as tmpdirname:
        print('created temporary directory', tmpdirname)
        output_dir_1 = Path(tmpdirname) / "one"
        output_dir_2 = Path(tmpdirname) / "two"
        output_dir_1.mkdir(exist_ok=True, parents=True)
        output_dir_2.mkdir(exist_ok=True, parents=True)

        extract_archive(path_to_archive_1, to=output_dir_1, tar_command=tar_command)
        extract_archive(path_to_archive_2, to=output_dir_2, tar_command=tar_command)

        matches = get_matching_file_pairs(output_dir_1, output_dir_2)
        print(matches)


        for path_1, path_2 in matches:
            subprocess.run(["meld", str(path_1), str(path_2)])
def get_matching_file_pairs(output_dir_1, output_dir_2):
    """Iterate over files in dir_1 and find best matching file in dir_2"""

    def pop_best_match(remaining_paths : List[Tuple[Path, str]], best_match : str) -> Tuple[Path, str]:
        for idx, item in enumerate(remaining_paths):
            if item[1] == best_match:
                return remaining_paths.pop(idx)


    remaining_paths : List[Tuple[Path, str]] = list(map(lambda p : (p, remove_date(str(p.name))), output_dir_2.iterdir()))
    matches : List[Tuple[Path, Path]]= []
    for path_1 in output_dir_1.iterdir():
        filename_1 = str(path_1.name)
        best_match, _ = process.extractOne(filename_1, map(lambda t: t[1], remaining_paths)) # compare strings
        best_match_path = pop_best_match(remaining_paths, best_match)[0]
        matches.append((path_1.absolute(), best_match_path.absolute()))
    return matches



if __name__ == "__main__":
    app()
