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

    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        raise NotImplementedError(f"Saving {type(data)} to csv not implemented.")
    now = datetime.now().strftime("%m-%d_%H:%M")
    filename = f"{name}_{now}_{USER}"
    FILENAMES.append(filename)
    df.to_csv(filename)
