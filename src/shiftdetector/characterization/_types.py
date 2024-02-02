# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from threading import Condition, Thread
from typing import TYPE_CHECKING

from typing_extensions import Self

if TYPE_CHECKING:
    import numpy as np


class AbstractMeasure(ABC):
    """
    Abstract class for data measurement.

    This class is an abstract class for data measurement. It is used to
    define the interface for data measurement classes.

    Methods
    -------
    start()
        Start the data measurement.
    stop()
        Stop the data measurement.
    get_data()
        Get the data data.
    measure_data()
        Measure the data. Abstract method.
    reset()
        Reset the data.
    """

    def __init__(self: Self, interval: float = 0.05) -> None:
        self._interval = interval
        self._thread = Thread(target=self._run, daemon=True)
        self._condition = Condition()
        self._data: list[float] = []
        self._stopped = False
        self._thread.start()

    def start(self: Self) -> None:
        """Start the data measurement."""
        with self._condition:
            self._stopped = False
            self._condition.notify()

    def stop(self: Self) -> None:
        """Stop the data measurement."""
        self._stopped = True
        self._thread.join()

    def get_data(self: Self) -> list[float]:
        """
        Get the data.

        Returns
        -------
        list[float]
            The data.
        """
        return self._data

    @abstractmethod
    def measure_data(self: Self) -> float:
        """
        Measure the data.

        Returns
        -------
        float
            The measurement.
        """
        err_msg = "Not implemented in abstract class."
        raise NotImplementedError(err_msg)

    def _run(self: Self) -> None:
        with self._condition:
            self._condition.wait()
        while not self._stopped:
            t0 = time.perf_counter()
            self._data.append(self.measure_data())
            time.sleep(self._interval - (time.perf_counter() - t0))

    def reset(self: Self) -> None:
        """Reset the data."""
        self._data = []


class AbstractModel(ABC):
    """
    Abstract class for representing a model for characterization.

    Methods
    -------
    preprocess()
        Preprocess the input data.
    predict()
        Run the model on the input data.
    __call__()
        Run the model on the input data.
    """

    @abstractmethod
    def __init__(self: Self) -> None:
        pass

    @abstractmethod
    @property
    def input_size(self: Self) -> tuple[int, int]:
        """
        Get the input size of the model.

        Returns
        -------
        tuple[int, int]
            The input size of the model.
        """
        err_msg = "Not implemented in abstract class."
        raise NotImplementedError(err_msg)

    @abstractmethod
    def preprocess(self: Self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess the input data.

        Parameters
        ----------
        data : np.ndarray
            The input data.

        Returns
        -------
        np.ndarray
            The preprocessed data.
        """
        err_msg = "Not implemented in abstract class."
        raise NotImplementedError(err_msg)

    @abstractmethod
    def predict(
        self: Self,
        data: np.ndarray,
    ) -> list[tuple[tuple[int, int, int, int], int, int]]:
        """
        Predict on the input data.

        Parameters
        ----------
        data : np.ndarray
            The input data.

        Returns
        -------
        list[tuple[int, tuple[int, int, int, int]]]
            The output data as a list containing the bounding box, class, and confidence score.
        """
        err_msg = "Not implemented in abstract class."
        raise NotImplementedError(err_msg)

    def __call__(
        self: Self,
        data: np.ndarray,
        *,
        preprocessed: bool | None = None,
    ) -> list[tuple[tuple[int, int, int, int], int, int]]:
        """
        Run the model on the input data.

        Parameters
        ----------
        data : np.ndarray
            The input data.
        preprocessed : bool, optional
            Whether the input data is preprocessed, by default None.
            If None, the input data will be preprocessed.

        Returns
        -------
        list[tuple[tuple[int, int, int, int], int, int]]
            The output data.
        """
        if preprocessed is None:
            preprocessed = False

        if not preprocessed:
            data = self.preprocess(data)

        return self.predict(data)
