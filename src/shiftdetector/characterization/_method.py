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
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import cv2

from shiftdetector.common import compute_map, scale_coords

from ._helpers import get_power_draw, get_steady_state

if TYPE_CHECKING:
    from ._types import AbstractMeasure, AbstractModel


def _characterize(
    model: AbstractModel,
    modelname: str,
    output_dir: Path,
    power_reader: AbstractMeasure,
    steady_state_power: float,
    image_dir: Path,
    image_names: list[str],
    ground_truth: list[list[tuple[tuple[int, int, int, int], int]]],
    num_power_iterations: int = 100,
    dummy_image_size: tuple[int, int, int] = (640, 480, 3),
    map_iou_threshold: float = 0.5,
    *,
    use_cached: bool | None = None,
) -> None:
    """
    Characterize the given model.

    This function characterizes the given model and saves the
    characterization to the given output directory.

    Parameters
    ----------
    model : AbstractModel
        The model to characterize.
        Output assumed to be in the x1, y1, x2, y2 format.
    modelname : str
        The name of the model to characterize.
    output_dir : Path
        The directory to save the characterization to.
    power_reader : AbstractMeasure
        The power reader to use for measuring power.
    steady_state_power : float
        The steady state power draw of the system.
    image_dir : Path
        The directory containing the images to use for characterization.
    image_names : list[str]
        The list of image names to use for characterization.
    ground_truth : list[list[tuple[tuple[int, int, int, int], int]]]
        The ground truth bounding boxes and classes for the images.
        Bounding boxes are assumed to be in the x1, y1, x2, y2 format.
    num_power_iterations : int, optional
        The number of power iterations to take, by default 100
    dummy_image_size : tuple[int, int, int], optional
        The size of the dummy image to use, by default (640, 480, 3)
    map_iou_threshold : float, optional
        The IoU threshold to use for computing mAP, by default 0.5
    use_cached : bool, optional
        Whether to use the cached characterization, by default None
        If None, will use the cached characterization if it exists.

    Raises
    ------
    FileNotFoundError
        If the image directory does not exist.
    """
    if use_cached is None:
        use_cached = True

    if not Path.exists(output_dir / modelname):
        Path.mkdir(output_dir / modelname, parents=True)

    jsonpath = output_dir / modelname / f"{modelname}.json"
    if not Path.exists(jsonpath) or not use_cached:
        energy = get_power_draw(
            model,
            power_reader,
            steady_state_power,
            num_power_iterations,
            dummy_image_size,
        )
        with Path.open(jsonpath, "w") as f:
            json.dump({"power_draw": energy}, f, ident=4)
    else:
        jsonstr: str = str(jsonpath.resolve())
        energy = float(json.load(jsonstr)["power_draw"])  # type: ignore[arg-type]

    if not Path.exists(image_dir):
        err_msg = f"Image directory {image_dir} does not exist."
        raise FileNotFoundError(err_msg)

    image_stats = defaultdict(defaultdict(dict))

    for image_name, gt in zip(image_names, ground_truth):
        imagepath = image_dir / image_name
        image = cv2.imread(str(imagepath))

        t0 = time.perf_counter()
        tensor = model.preprocess(image)
        t1 = time.perf_counter()
        outputs = model(tensor, preprocessed=True)
        t2 = time.perf_counter()
        t_preprocess = t1 - t0
        t_inference = t2 - t1

        outputs = [
            (scale_coords(bbox, image.shape[:2], model.input_size), class_id, score) for bbox, class_id, score in outputs
        ]
        accuracy, iou, conf = compute_map(gt, outputs, iou_threshold=map_iou_threshold)

        image_stats[image_name]["preprocess"] = t_preprocess
        image_stats[image_name]["time"] = t_inference
        image_stats[image_name]["accuracy"] = accuracy
        image_stats[image_name]["iou"] = iou
        image_stats[image_name]["conf"] = conf
        image_stats[image_name]["energy"] = t_inference * energy


def characterize(
    model_funcs: list[Callable[[], AbstractModel]],
    model_names: list[str],
    output_dir: Path,
    power_reader: AbstractMeasure,
    get_memory: Callable[[], tuple[int, int, int]],
    num_power_iterations: int = 100,
    dummy_image_size: tuple[int, int, int] = (640, 480, 3),
    *,
    characterize_models: bool | None = None,
    use_cached_energy: bool | None = None,
) -> None:
    """
    Characterize the given models.

    This function characterizes the given models and saves the
    characterization to the given output directory.

    Parameters
    ----------
    model_funcs : list[Callable[[], type[AbstractModel]]]
        The list of model functions to characterize.
        Each function should return an instance of a model.
    model_names : list[str]
        The list of model names to use for the models.
    output_dir : Path
        The directory to save the characterization to.
    power_reader : AbstractMeasure
        The power reader to use for measuring power.
    get_memory : Callable[[], tuple[int, int, int]]
        The function to use for getting memory usage.
        Gets the total memory, free memory, and used memory.
    num_power_iterations : int, optional
        The number of power iterations to take, by default 100
    dummy_image_size : tuple[int, int, int], optional
        The size of the dummy image to use, by default (640, 480, 3)
    characterize_models : bool, optional
        Whether to characterize the models, by default None
        If None, will set to True and will characterize the
        models if the output directory does not contain a
        characterization.
    use_cached_energy : bool, optional
        Whether to use the cached energy values, by default None
        If None, will use the cached energy values if they exist.
    """
    if characterize_models is None:
        characterize_models = True

    if not Path.exists(output_dir):
        Path.mkdir(output_dir, parents=True)
        characterize_models = True

    jsonpath = output_dir / "steady_state.json"
    if not Path.exists(jsonpath):
        steady_state = get_steady_state(power_reader, get_memory)
        with Path.open(jsonpath, "w") as f:
            json.dump(steady_state, f, ident=4)

    jsonstr = str(jsonpath.resolve())
    steady_state = json.load(jsonstr)  # type: ignore[arg-type]
    steady_state_power = float(steady_state["power_draw"])

    if characterize_models:
        for model_func, model_name in zip(model_funcs, model_names):
            model = model_func()
            _characterize(
                model=model,
                modelname=model_name,
                output_dir=output_dir,
                power_reader=power_reader,
                steady_state_power=steady_state_power,
                num_power_iterations=num_power_iterations,
                dummy_image_size=dummy_image_size,
                use_cached=use_cached_energy,
            )
            del model

