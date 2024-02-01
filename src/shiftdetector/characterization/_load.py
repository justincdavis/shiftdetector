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

from pathlib import Path
from typing import Callable, TYPE_CHECKING

from ._types import AbstractModel, AbstractMeasure

if TYPE_CHECKING:
    pass

def _measure_load(
    output_dir: Path,
    model_func: Callable[[], AbstractModel],
    modelname: str,
    power_reader: AbstractMeasure,
    memory_func: Callable[[], tuple[int, int, int]],
    import_handle: Callable[[], None],
    num_iterations: int = 20,
) -> None:
    """
    Measure the load time, energy, and memory usage of a model.

    Parameters
    ----------
    output_dir : Path
        The directory to save the measurements to.
    model_func : Callable[[], AbstractModel]
        The function to create the model.
    modelname : str
        The name of the model.
    power_reader : AbstractMeasure
        The power reader to use.
    memory_func : Callable[[], tuple[int, int, int]]
        The function to get the memory usage.
        Returns the total memory, free memory, and used memory.
    import_handle : Callable[[], None]
        The function to import the model.
    num_iterations : int, optional
        The number of iterations to measure, by default 20
    """
    import_handle()
    total_mem, free_mem, used_mem = memory_func()

    load_times: list[float] = []
    load_energy: list[float] = []
    load_mem_free: list[float] = []
    load_mem_used: list[float] = []
    load_data = {
        "energy": {},
        "time": {},
        "mem": {},
    }

    for _ in range(num_iterations):
        