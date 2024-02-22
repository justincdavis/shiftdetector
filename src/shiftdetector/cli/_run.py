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

import csv
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
from cv2ext import IterableVideo

if TYPE_CHECKING:
    from shiftdetector.characterization import AbstractMeasure, AbstractModel


def run(
    videofile: Path,
    output_dir: Path,
    modelname: str,
    modelfunc: Callable[[], AbstractModel],
    power_measure: AbstractMeasure | None = None,
    power_draw: float | None = None,
    characterization_dir: Path | None = None,
) -> None:
    """
    Run a model on a video file.

    Parameters
    ----------
    videofile : Path
        The path to the video file to process.
    output_dir : Path
        The directory to save the results.
    modelname : str
        The name of the model.
    modelfunc : Callable[[], AbstractModel]
        A function that returns a model instance.
    power_measure : AbstractMeasure | None
        An optional power measure implementation to use.
        If not provided, the energy usage will be determined from
        the supplied power_draw parameter.
        If None, power_draw must be provided.
    power_draw : float | None
        An optional power draw value to use.
        If not provided, the energy usage will be determined from
        the supplied power_measure parameter.
        If None, power_measure must be provided.
    characterization_dir : Path | None
        An optional directory to load the characterization data from.
        When specified and power_measure/power_draw are None,
        the power draw will be determined from the characterization data.

    Raises
    ------
    ValueError
        If neither power_measure nor power_draw is provided.
    FileNotFoundError
        If the characterization data for the model does not exist.

    """
    if (
        power_measure is None
        and power_draw is None
        and characterization_dir is not None
    ):
        err_msg = "Either power_measure, power_draw, or characterization_dir must be provided."
        raise ValueError(err_msg)
    if (
        power_measure is None
        and power_draw is None
        and characterization_dir is not None
    ):
        jsonpath = characterization_dir / modelname / f"{modelname}.json"
        if not Path.exists(jsonpath):
            err_msg = f"The characterization data for {modelname} does not exist at {jsonpath}."
            raise FileNotFoundError(err_msg)
        with Path.open(jsonpath, "w") as f:
            data = json.load(f)
            power_draw = float(data["power_draw"])

    results: list[tuple[int, int, int, int, int, float, float, float, float, float]] = (
        []
    )

    model = modelfunc()
    for frameid, frame in IterableVideo(videofile):
        t0 = time.perf_counter()
        tensor = model.preprocess(frame)
        t1 = time.perf_counter()

        if power_measure is not None:
            power_measure.start()

        t2 = time.perf_counter()
        bbox, score = model.predict(tensor)
        t3 = time.perf_counter()

        if power_measure is not None:
            power_measure.stop()
            power = float(np.mean(power_measure.get_data()))
        else:
            # power_draw will be defined if power_measure is None
            power = power_draw  # type: ignore[assignment]

        energy = power * (t3 - t2)
        preprocess_time = t1 - t0
        latency = t3 - t2

        results.append((frameid, *bbox, score, energy, power, preprocess_time, latency))
    del model

    videoname = videofile.stem
    output_file = output_dir / f"{modelname}_{videoname}.csv"

    with Path.open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "frameid",
                "x1",
                "y1",
                "x2",
                "y2",
                "score",
                "energy",
                "power",
                "preprocess",
                "latency",
            ],
        )
        writer.writerows(results)
