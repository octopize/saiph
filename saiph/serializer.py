import json
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from saiph.models import Model


class ModelJSONSerializer:

    # !Make sure to update the version if you change NumpyPandasEncoder or ModelJSONSerializer
    VERSION = "1.0"

    def encode(self, coords: NDArray[np.float_], model: Model) -> Tuple[str, str]:

        coords_encoding = {"data": coords, "__version__": self.VERSION}
        encoded_coords = json.dumps(coords_encoding, cls=NumpyPandasEncoder)

        encoding = {"data": model.__dict__, "__version__": self.VERSION}
        encoded_model = json.dumps(encoding, cls=NumpyPandasEncoder)
        return encoded_coords, encoded_model

    def decode(
        self, raw_coords: str, raw_model: str
    ) -> Tuple[NDArray[np.float_], Model]:
        coords = json.loads(raw_coords, object_hook=numpy_pandas_json_obj_hook)
        model_dict = json.loads(raw_model, object_hook=numpy_pandas_json_obj_hook)
        return coords["data"], Model(**model_dict["data"])


class NumpyPandasEncoder(json.JSONEncoder):
    def default(self, obj):
        """Encode numpy arrays, pandas dataframes, and pandas series, or objects containing them.

        :param obj: object to encode
        """
        if isinstance(obj, np.ndarray):
            data = obj.tolist()
            return dict(__ndarray__=data, dtype=str(obj.dtype), shape=obj.shape)

        if isinstance(obj, pd.Series):
            data = obj.to_json(orient="index", default_handler=str)
            return dict(__series__=data, dtype=str(obj.dtype))

        if isinstance(obj, pd.DataFrame):
            # orient='table' includes dtypes but doesn't work
            data = obj.to_json(orient="index", default_handler=str)
            return dict(__frame__=data)

        super().default(obj)


def numpy_pandas_json_obj_hook(json_dict):
    """Decode numpy arrays, pandas dataframes, and pandas series, or objects containing them.

    :param json_dict: (dict) json encoded model object
    """
    # Numpy arrays
    if isinstance(json_dict, dict) and "__ndarray__" in json_dict:
        data = json_dict["__ndarray__"]
        return np.asarray(data, dtype=json_dict["dtype"]).reshape(json_dict["shape"])

    # Pandas Series
    if isinstance(json_dict, dict) and "__series__" in json_dict:
        data = json_dict["__series__"]
        return pd.read_json(data, orient="index", typ="series").astype(
            json_dict["dtype"]
        )

    # Pandas dataframes
    if isinstance(json_dict, dict) and "__frame__" in json_dict:
        data = json_dict["__frame__"]
        # We could include dtypes, but ATM we have columns with integers
        # labels (0, 1, 2, ..), and JSON transforms them to string
        # with makes converting the dtypes not possible ('0' != 0).
        # Omitting the dtype still provides the right dtype.
        return pd.read_json(data, orient="index", typ="frame")

    return json_dict
