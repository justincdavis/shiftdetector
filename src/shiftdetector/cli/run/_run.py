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
# ruff: noqa: PLC0415, I001
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np
from cv2ext import IterableVideo
from tqdm import tqdm

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
    *,
    progress_bar: bool | None = None,
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
    progress_bar : bool | None
        An optional flag to display a progress bar.
        If None, then no progress bar will be displayed.

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
    if progress_bar is None:
        progress_bar = False

    results: list[tuple[int, int, int, int, int, float, float, float, float, float]] = (
        []
    )

    video: Iterable[tuple[int, np.ndarray]] = IterableVideo(videofile)
    if progress_bar:
        video = tqdm(video)

    model = modelfunc()
    for frameid, frame in video:
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


def _get_oakd_model(
    blobpath: str,
    input_size: tuple[int, int],
) -> Callable[[], AbstractModel]:
    import shiftdetector.implementations.oakd as oakd  # type: ignore[import-not-found]

    def _model() -> AbstractModel:
        return oakd.OakModel(blobpath, input_size=input_size)  # type: ignore[no-any-return]

    return _model


def _get_yolo_tensorrt_model(modelpath: str) -> Callable[[], AbstractModel]:
    import shiftdetector.implementations.tensorrt as trt  # type: ignore[import-not-found]

    def _model() -> AbstractModel:
        return trt.YoloModel(modelpath)  # type: ignore[no-any-return]

    return _model


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Run a model on a video.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model.")
    parser.add_argument("--name", type=str, required=False, help="Name of the model.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--input_size",
        type=str,
        required=False,
        help="Size of the model input. Should be of form: [124,124]",
    )
    parser.add_argument(
        "--platform",
        type=str,
        required=False,
        help="Platform to run the model on.",
    )
    parser.add_argument(
        "--powerdraw",
        type=float,
        required=False,
        help="Power draw of the model.",
    )
    parser.add_argument(
        "--char_dir",
        type=str,
        required=False,
        help="Path to the characterization directory.",
    )
    args = parser.parse_args()

    modelname = args.name if args.name else Path(args.model).stem

    # parse input size
    input_size = None
    if args.input_size is not None:
        width, height = tuple(map(int, args.input_size.strip("[]").split(",")))
        input_size = (width, height)

    # handle modelfunc
    extension = Path(args.model).suffix
    modelfunc = None
    if extension == ".blob":
        if input_size is None:
            err_msg = "Input size must be provided for OAK-D models."
            raise ValueError(err_msg)
        modelfunc = _get_oakd_model(args.model, input_size)
    elif extension == ".engine" or extension == ".trt":
        if "yolo" in modelname.lower():
            modelfunc = _get_yolo_tensorrt_model(args.model)
        else:
            err_msg = f"Do not have an implementation available for model {modelname},"
            err_msg = " using TensorRT. Currently only YoloV7 is supported."
            raise ValueError(err_msg)

    # handle power_measure
    power_measure = None
    if args.platform is not None:
        if args.platform == "jetson":
            import shiftdetector.implementations.jetson as jt  # type: ignore[import-not-found]

            power_measure = jt.JetsonMeasure()
        else:
            err_msg = (
                f"Do not have an implementation available for platform {args.platform}."
            )
            raise ValueError(err_msg)

    if modelfunc is None:
        err_msg = f"Do not have an implementation available for model {modelname}."
        raise ValueError(err_msg)

    run(
        videofile=args.video,
        output_dir=args.output_dir,
        modelname=args.model,
        modelfunc=modelfunc,
        power_measure=power_measure,
        power_draw=args.powerdraw,
        characterization_dir=args.char_dir,
        progress_bar=True,
    )
