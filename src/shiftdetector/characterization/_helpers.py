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
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from ._types import AbstractMeasure, AbstractModel


def get_steady_state(
    power_reader: AbstractMeasure,
    get_memory: Callable[[], tuple[int, int, int]],
    sample_time: float = 10.0,
) -> dict[str, str]:
    """
    Get the steady state power and memory usage of the system.

    Parameters
    ----------
    power_reader : AbstractMeasure
        The power reader to use.
    get_memory : Callable[[], tuple[int, int]]
        The function to get the memory usage.
    sample_time : float, optional
        The time to sample the power and memory, by default 10.0 seconds

    Returns
    -------
    dict[str, str]
        The steady state power and memory usage.
        Keys are "power_draw", "total_mem", "free_mem", and "used_mem".
    """
    # sample power
    power_reader.start()
    time.sleep(sample_time)
    power_reader.stop()
    power_readings = power_reader.get_data()

    # sample memory
    total_memory = 0.0
    free_memory_readings = []
    used_memory_readings = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sample_time:
        total_memory, free_memory, used_memory = get_memory()
        free_memory_readings.append(free_memory)
        used_memory_readings.append(used_memory)

    return {
        "power_draw": str(np.mean(power_readings)),
        "total_mem": str(total_memory),
        "free_mem": str(np.mean(free_memory_readings)),
        "used_mem": str(np.mean(used_memory_readings)),
    }


def get_power_draw(
    model: AbstractModel,
    power_reader: AbstractMeasure,
    steady_state: float,
    num_iterations: int = 100,
    image_size: tuple[int, int, int] = (640, 480, 3),
) -> float:
    """
    Get the power draw of the given model.

    Parameters
    ----------
    model : AbstractModel
        The model to get the power draw of.
    power_reader : AbstractMeasure
        The power reader to use.
    steady_state : float
        The steady state power draw.
    num_iterations : int, optional
        The number of iterations to run the model, by default 100
    image_size : tuple[int, int, int], optional
        The size of the image to use for the model, by default (640, 480, 3)
        All models should have their own preprocess method that takes an image
        and returns a tensor. This method is used to preprocess the image
        before passing it to the model.

    Returns
    -------
    float
        The power draw of the model.
    """
    dummy_image: np.ndarray = np.ones(image_size, dtype=np.uint8)
    tensor = model.preprocess(dummy_image)
    energy_vals = []
    power_reader.start()
    for _ in range(num_iterations):
        model(tensor, preprocessed=True)
    power_reader.stop()
    readings = power_reader.get_data()
    energy_vals = [reading - steady_state for reading in readings]
    return float(np.mean(energy_vals))
