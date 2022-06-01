import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Any, Union
from datetime import datetime

USER = "UNCONFIGURED"
FILENAMES = []
def set_active_user(user : str):
    global USER
    USER = user

def get_filenames():
    return FILENAMES

def to_csv(data : Union[pd.DataFrame, NDArray[Any]], name : str):
    global FILENAMES

    now = datetime.now().strftime("%m-%d_%H:%M")
    filename = f"{name}_{now}_{USER}"

    if isinstance(data, pd.DataFrame):

        with open(filename, "a+") as f:
            f.write(f"{data.shape} items \n")
            for col in data.columns:
                f.write("Column name : " + str(col) + "\n")
                f.write(data[col].to_string(max_rows=None))
    elif isinstance(data, np.ndarray):
        with open(filename, "w") as f:
            arr = np.array2string(data, separator='\n', suppress_small=True, threshold=10000000000)
            f.write(arr)
    else:
        raise NotImplementedError(f"Saving {type(data)} to file not implemented.")

    FILENAMES.append(filename)
