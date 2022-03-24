from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from saiph.models import Model
from saiph.serializer import AbstractSerializer


class AbstractBackend:
    def __init__(self, serializer: AbstractSerializer):
        self.serializer = serializer

    def load(self) -> Tuple[NDArray[np.float_], Model]:
        pass

    def save(coords: NDArray[np.float_], model: Model) -> None:
        pass


class DiskBackend(AbstractBackend):
    def __init__(
        self,
        serializer: AbstractSerializer,
        coords_filename: Path,
        model_filename: Path,
    ) -> None:
        super().__init__(serializer)
        self.coords_filename = coords_filename
        self.model_filename = model_filename

    def load(self) -> Tuple[NDArray[np.float_], Model]:
        with open(self.coords_filename, "rb") as file:
            raw_coords = file.read(self.coords_filename)
        with open(self.model_filename, "rb") as file:
            raw_model = file.read(self.model_filename)

        return self.serializer.decode(raw_coords, raw_model)

    def save(self, coords: NDArray[np.float_], model: Model) -> None:
        encoded_coords, encoded_model = self.serializer.encode(coords, model)
        with open(self.coords_filename, "wb") as file:
            file.write(encoded_coords)
        with open(self.model_filename, "wb") as file:
            file.write(encoded_model)
