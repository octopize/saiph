#!/usr/bin/env python3

import csv
import os
import random
import sys
from pathlib import Path
from typing import Any

import typer
from faker import Faker
from tqdm import tqdm  # type: ignore

fake = Faker("en_US")
CITIES = [
    "Lyon",
    "Nantes",
    "Paris",
    "Marseille",
    "Cholet",
    "Rochefourchat",
    "Ornes",
    "Senconac",
    "Caunette-sur-Lauquet",
    "Cherbourg",
]
CITIES_WEIGHTS = (20, 10, 50, 10, 4, 1, 1, 1, 1, 2)


def eprint(*args: Any, **kwargs: dict[str, Any]) -> None:
    print(*args, file=sys.stderr, **kwargs)  # noqa: T201 # type: ignore


def human_size(size: int, units: list[str] = ["bytes", "KB", "MB", "GB", "TB", "PB", "EB"]) -> str:
    """Return a human readable string representation of bytes."""
    return (
        str(size) + units[0]
        if size < 1024 or len(units) == 1
        else human_size(size >> 10, units[1:])
    )


def get_city() -> str:
    return random.choices(CITIES, weights=CITIES_WEIGHTS)[0]  # noqa: S311


def get_row(dimension_count: int = 3) -> dict[str, Any]:
    r = {
        "latitude": fake.latitude(),
        "longitude": fake.longitude(),
        "city": get_city(),
    }
    r.update({f"city_{i}": get_city() for i in range(dimension_count - 3)})

    return r


def main(
    outfile: Path,
    row_count: int = 1000,
    dimension_count: int = 3,
) -> int:
    """Generate CSV with fake data."""
    eprint(f"generating {row_count} fake rows")

    if dimension_count < 3:
        raise ValueError(f"Expected dimension_count >= 3, got {dimension_count} instead")

    header = list(get_row(dimension_count).keys())
    with open(outfile, "w") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for i in tqdm(range(row_count)):
            writer.writerow(get_row(dimension_count))

    if Path(outfile.name).exists():
        size = os.path.getsize(outfile.name)
        eprint(f"Wrote {human_size(size)} file")

    return 0


if __name__ == "__main__":
    typer.run(main)
