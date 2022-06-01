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
from saiph.conftest import _wbcd_csv, _wbcd_supplemental_coordinates_csv_mca, _wbcd_supplemental_mitoses_first, _wbcd_mitoses_first
from saiph.test_utils import get_filenames, set_debug_mode, to_file, set_active_user
import subprocess
import platform
from matplotlib import pyplot as plt

import tempfile 

from thefuzz import process

app = typer.Typer()

REVERSED_NAME = "reversed_individuals"
def get_original_df():
    df = _wbcd_mitoses_first.copy()
    if "Sample_code_number" in df.columns:
        df = df.drop(columns=["Sample_code_number"])
    return df.astype("category")

def get_reversed_df(df):
    coordinates = _wbcd_supplemental_mitoses_first.copy()
    model = fit(df, nf="all")
    reversed_individuals = inverse_transform(coordinates, model)
    return reversed_individuals.astype("int")

@app.command("generate")
def generate_debug_files(user : str, archive_name: str =typer.Argument(""), compress : bool = True):

    set_active_user(user)
    set_debug_mode(True)

    original = get_original_df()
    reversed_individuals = get_reversed_df(original)

    to_file(reversed_individuals, REVERSED_NAME)

    if compress:
        return generate_archive(user, filename=archive_name)
    
    return 0

@app.command("explore")
def build_histogram():
    set_debug_mode(False)

    original = get_original_df()
    reversed = get_reversed_df(original)
    
    original = original.astype("int")

    grid_size = ceil(sqrt(len(original.columns)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(20, 20))

    for ax, col in zip(axs.flat, original.columns):
        original.hist(col, ax=ax , alpha=0.5, label="Original")
        reversed.hist(col, ax=ax,alpha=0.5, label = "Inverse")
        ax.legend()

    fig.suptitle(f"Inverse transform on {platform.system()} { platform.machine()}", fontsize=20)

    filename = f"distributions_{platform.system()}.svg"
    fig.savefig(filename)

    print("Saving figure to ", filename)

    plt.show()

@app.command("compare_histogram")
def compare_histogram(path_to_archive_1 : Path, path_to_archive_2 : Path):
    """Compare the generated histogram from current machine to one saved in a file"""
    set_debug_mode(False)

    original = get_original_df()
    results_1 = get_reversed_from_archive(path_to_archive_1)
    print(results_1.columns)
    results_2 = get_reversed_from_archive(path_to_archive_2)
    print(results_2.columns)
    grid_size = ceil(sqrt(len(original.columns)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(20, 20))

    for ax, col in zip(axs.flat, original.columns):
        results_1.hist(col, ax=ax,alpha=0.5, label = "Archive 1",color="magenta")
        results_2.hist(col, ax=ax , alpha=0.5, label = "Archive 2", color="lime")
        ax.legend()

    fig.suptitle(f"Comparison of reversed individuals from two archives.", fontsize=20)

    filename = f"comparison_of_distributions_{platform.system()}.svg"
    fig.savefig(filename)

    print("Saving figure to ", filename)

    plt.show()

def get_reversed_from_archive(path_to_archive : Path) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmpdirname:
        extract_archive(path_to_archive, to=tmpdirname)
        filename : Path
        for file in Path(tmpdirname).iterdir():
            if str(file.name).startswith(REVERSED_NAME) and not str(file.name).endswith("flattened") :
                filename = file
                print("Found filename in archive : ", filename)
                break

        if not filename:
            raise ValueError(f"Expected to find a file starting with {REVERSED_NAME}, got nothing. "
            "Maybe the filename changed ?")
    
        return pd.read_csv(filename, index_col=0)


def generate_archive(name : str, filename : str = ""):
    """Generate an archive of the files and delete them after. """
    now = datetime.now().strftime("%m-%d_%Hh%M")

    if filename and not filename.endswith(".tar.gz"):
        filename += ".tar.gz"
    archive_name = filename if filename else f"debug_result_{name}_{now}.tar.gz"

    process = subprocess.run([get_tar_command(), "czf", archive_name, *get_filenames(), "--remove-files"]) 

    if process.returncode == 0:
        print(f"Success! Created archive '{filename}'")
        return 0
    print("FAILED!")

    return 1

def extract_archive(archive : Path, to : Path):
    process = subprocess.run([get_tar_command(), "xzf", archive, f"--directory={to}"]) 
    if process.returncode != 0:
        raise Exception("There was an ERROR. Exiting ...")
    
    print(f"Extracted archive '{archive.name}' to {to}")

def remove_date(filename : str) -> str:
    s = re.sub("\d", repl="", string=filename)
    s = re.sub("-_:", repl="", string=s)
    return s

def get_tar_command() -> str:
    tar_command = "tar" if platform.system() == "Linux" else "gtar"

    if tar_command != "tar":
        print("This program uses GNU tar. If you are on Mac, you can install it using 'brew install gnu-tar'.")
        input("Press Enter to continue if you have installed it...")
    return tar_command

@app.command("compare")
def compare(path_to_archive_1 : Path, path_to_archive_2 : Path):

    print("This program uses 'meld' to compare files. Make sure you have it installed.\n")
    input("Press Enter to continue if you have installed it...")



    with tempfile.TemporaryDirectory() as tmpdirname:
        print('created temporary directory', tmpdirname)
        output_dir_1 = Path(tmpdirname) / "one"
        output_dir_2 = Path(tmpdirname) / "two"
        output_dir_1.mkdir(exist_ok=True, parents=True)
        output_dir_2.mkdir(exist_ok=True, parents=True)

        extract_archive(path_to_archive_1, to=output_dir_1)
        extract_archive(path_to_archive_2, to=output_dir_2)

        matches = get_matching_file_pairs(output_dir_1, output_dir_2)

        for path_1, path_2 in matches:
            subprocess.run(["meld", str(path_1), str(path_2)])
    
    return 0

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
