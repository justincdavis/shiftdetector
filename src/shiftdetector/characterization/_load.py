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

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from ._helpers import get_steady_state

if TYPE_CHECKING:
    from ._types import AbstractMeasure, AbstractModel


def measure_load(
    output_dir: Path,
    model_func: Callable[[], AbstractModel],
    modelname: str,
    power_reader: AbstractMeasure,
    memory_func: Callable[[], tuple[int, int, int]],
    import_handle: Callable[[], None],
    num_iterations: int = 20,
    steady_state_sample_time: float = 10,
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
    steady_state_sample_time : float, optional
        The time to sample the steady state power and memory, by default 10.0 seconds
    """
    import_handle()

    steady_state_data = get_steady_state(
        power_reader,
        memory_func,
        steady_state_sample_time,
    )

    load_times: list[float] = []
    load_energy: list[float] = []
    load_mem_free: list[float] = []
    load_mem_used: list[float] = []
    load_data: dict[str, dict[str, str]] = {
        "energy": {},
        "time": {},
        "mem": {},
    }

    power_reader.start()
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        m = model_func()
        t1 = time.perf_counter()
        load_energy.extend(power_reader.get_data())
        load_times.append(t1 - t0)
        _, free_mem, used_mem = memory_func()
        load_mem_free.append(free_mem)
        load_mem_used.append(used_mem)
        del m
        power_reader.reset()
    power_reader.stop()

    steady_state_power = float(steady_state_data["power_draw"])
    steady_state_freemem = float(steady_state_data["free_mem"])
    steady_state_usedmem = float(steady_state_data["used_mem"])

    load_energy = [(e - steady_state_power) / 1000 for e in load_energy]
    # how much less free memory compared to steady state
    free_mem_vals = [steady_state_freemem - f for f in load_mem_free]
    # how much more used memory compared to steady state
    used_mem_vals = [u - steady_state_usedmem for u in load_mem_used]
    # model mem is average of less free mem and more used mem
    model_mem = [(f + u) / 2 for f, u in zip(free_mem_vals, used_mem_vals)]
    for n, d in zip(["energy", "time", "mem"], [load_energy, load_times, model_mem]):
        load_data[n]["mean"] = str(np.mean(d))
        load_data[n]["median"] = str(np.median(d))
        load_data[n]["var"] = str(np.var(d))
        load_data[n]["std"] = str(np.std(d))
        load_data[n]["min"] = str(np.min(d))
        load_data[n]["max"] = str(np.max(d))

    json_path = Path(output_dir, modelname, f"{modelname}.json")
    if Path.exists(json_path):
        with Path.open(json_path, "r") as f:
            json_data = json.load(f)
    else:
        json_data = {}

    json_data["loading"] = load_data
    with Path.open(json_path, "w") as f:
        json.dump(json_data, f, ident=4)
