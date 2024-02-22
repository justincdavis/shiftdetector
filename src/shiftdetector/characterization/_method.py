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
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import cv2  # type: ignore[import-untyped]
import numpy as np

from shiftdetector.common import bbox_iou, scale_coords

from ._graph import build_graph
from ._helpers import get_power_draw, get_steady_state
from ._load import measure_load

if TYPE_CHECKING:
    import networkx as nx  # type: ignore[import-untyped]

    from ._types import AbstractMeasure, AbstractModel


def _characterize(
    model: AbstractModel,
    modelname: str,
    output_dir: Path,
    power_reader: AbstractMeasure,
    steady_state_power: float,
    image_dir: Path,
    image_names: list[str],
    ground_truth: list[tuple[int, int, int, int]],
    num_power_iterations: int = 100,
    dummy_image_size: tuple[int, int, int] = (640, 480, 3),
    num_bins: int = 10,
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
    ground_truth : list[tuple[tuple[int, int, int, int], int]]
        The ground truth bounding boxes and classes for the images.
        Bounding boxes are assumed to be in the x1, y1, x2, y2 format.
    num_power_iterations : int, optional
        The number of power iterations to take, by default 100
    dummy_image_size : tuple[int, int, int], optional
        The size of the dummy image to use, by default (640, 480, 3)
    num_bins : int, optional
        The number of bins to use for binning the data, by default 10
    use_cached : bool, optional
        Whether to use the cached characterization, by default None
        If None, will use the cached characterization if it exists.

    Raises
    ------
    FileNotFoundError
        If the image directory does not exist.
    RuntimeError
        If bins and indices cannot be computed.
        If confidence value is greater than bin key.
        If confidence value is less than bin key.

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

    image_stats: dict[str, dict[str, float]] = defaultdict(dict)

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

        bbox, score = outputs[0], outputs[1]
        try:
            height, width, _ = image.shape  # type: ignore[misc]
        except ValueError:
            height, width = image.shape
        bbox = scale_coords(bbox, model.input_size, (width, height))
        iou = bbox_iou(bbox, gt)

        image_stats[image_name]["preprocess"] = t_preprocess
        image_stats[image_name]["time"] = t_inference
        image_stats[image_name]["accuracy"] = iou
        image_stats[image_name]["iou"] = iou
        image_stats[image_name]["conf"] = score
        image_stats[image_name]["energy"] = t_inference * energy

    fieldnames = list(image_stats[image_names[0]].keys())

    with Path.open(output_dir / modelname / f"{modelname}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for image_name, image_data in image_stats.items():
            entry = {"filename": image_name, **image_data}
            writer.writerow(entry)

    with Path.open(jsonpath, "r") as f:
        json_data = json.load(f)

    # raw metrics
    json_data["power_draw"] = str(energy)
    for metric in fieldnames:
        raw_data = [data[metric] for data in image_stats.values()]
        json_data[metric] = {
            "mean": str(np.mean(raw_data)),
            "median": str(np.median(raw_data)),
            "var": str(np.var(raw_data)),
            "std": str(np.std(raw_data)),
            "min": str(np.min(raw_data)),
            "max": str(np.max(raw_data)),
        }

    # correlations
    conf_data = np.array([data["conf"] for data in image_stats.values()])
    iou_data = np.array([data["iou"] for data in image_stats.values()])
    json_data["conf_corr"] = str(np.mean(np.corrcoef(conf_data, iou_data)))

    # binning
    json_bin_data = {}
    json_bin_data["num_bins"] = num_bins
    bin_array = np.array([b / num_bins for b in range(num_bins + 1)])
    indices = np.digitize(conf_data, bin_array)

    if len(indices) != len(conf_data):
        err_msg = "Could not compute bins and indices."
        raise RuntimeError(err_msg)

    binned_iou = defaultdict(list)
    binned_conf = defaultdict(list)

    for idx, indice in enumerate(indices):
        conf_key = indice / num_bins
        iou_val = iou_data[idx]
        conf_val = conf_data[idx]
        if conf_val > conf_key:
            err_msg = "Confidence value is greater than bin key."
            raise RuntimeError(err_msg)
        if conf_val < conf_key - 1 / num_bins:
            err_msg = "Confidence value is less than bin key."
            raise RuntimeError(err_msg)
        binned_iou[conf_key].append(iou_val)
        binned_conf[conf_key].append(conf_val)

    possible_bins = [b / num_bins for b in range(num_bins + 1)]
    for conf_bin in possible_bins:
        iou_bin_data = binned_iou[conf_bin]
        conf_bin_data = binned_conf[conf_bin]
        if len(iou_bin_data) == 0:
            iou_bin_data = [0.0, 0.0]
            conf_bin_data = [0.0, 0.0]
        alpha = np.mean(
            [i / c for i, c in zip(iou_bin_data, conf_bin_data) if c != 0],
        )
        if str(alpha) == "nan":
            alpha = 1.0
        z = np.array([1.0, 0.0])
        if not (np.sum(conf_bin_data) == 0 or np.sum(iou_bin_data) == 0):
            z = np.polyfit(conf_bin_data, iou_bin_data, 1)
        conf_bin_corr = float(np.mean(np.corrcoef(conf_bin_data, iou_bin_data)))
        if str(conf_bin_corr) == "nan":
            conf_bin_corr = 1.0
        sub_bin_data = {}
        sub_bin_data["alpha"] = str(alpha)
        sub_bin_data["conf_corr"] = str(conf_bin_corr)
        sub_bin_data["iou_mean"] = str(np.mean(iou_bin_data))
        sub_bin_data["conf_mean"] = str(np.mean(conf_bin_data))
        sub_bin_data["fit"] = str(z.tolist())
        # ignore type error here, since we want a dict with an int entry and
        # a dict entry so we can read int then read the str keys of the dict
        json_bin_data[str(conf_bin)] = sub_bin_data  # type: ignore[assignment]
    json_data["bins"] = json_bin_data

    # write final json file
    with Path.open(jsonpath, "w") as f:
        json.dump(json_data, f, ident=4)


def characterize(
    model_funcs: list[Callable[[], AbstractModel]],
    model_names: list[str],
    import_funcs: list[Callable[[], None]],
    output_dir: Path,
    power_reader: AbstractMeasure,
    get_memory: Callable[[], tuple[int, int, int]],
    image_dir: Path,
    image_files: list[str],
    ground_truth: list[tuple[int, int, int, int]],
    num_power_iterations: int = 100,
    steady_state_sample_time: float = 10.0,
    dummy_image_size: tuple[int, int, int] = (640, 480, 3),
    num_bins: int = 10,
    graph_metric: str = "conf",
    min_connects: int = 1,
    cutoff: float = 0.0,
    upper_outlier_cutoff: float = 80.0,
    connectivity: int = 2,
    graph_type: Callable[[], nx.Graph | nx.DiGraph] | None = None,
    *,
    characterize_models: bool | None = None,
    use_cached_energy: bool | None = None,
    create_graph: bool | None = None,
    purge_connectivity: bool | None = None,
    measure_load_stats: bool | None = None,
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
    import_funcs : list[Callable[[], None]]
        The list of import functions to use for the models.
        Each function should import all the necessary modules
        for and dependencies of the model.
    output_dir : Path
        The directory to save the characterization to.
    power_reader : AbstractMeasure
        The power reader to use for measuring power.
    get_memory : Callable[[], tuple[int, int, int]]
        The function to use for getting memory usage.
        Gets the total memory, free memory, and used memory.
    image_dir : Path
        The directory containing the images to use for characterization.
    image_files : list[str]
        The list of image names to use for characterization.
    ground_truth : list[list[tuple[tuple[int, int, int, int], int]]]
        The ground truth bounding boxes and classes for the images.
        Bounding boxes are assumed to be in the x1, y1, x2, y2 format.
    num_power_iterations : int, optional
        The number of power iterations to take, by default 100
    steady_state_sample_time : float, optional
        The time to sample the steady state power and memory, by default 10.0 seconds
    dummy_image_size : tuple[int, int, int], optional
        The size of the dummy image to use, by default (640, 480, 3)
    num_bins : int, optional
        The number of bins to use for binning the data, by default 10
    graph_metric : str, optional
        The metric to use for the graph, by default "conf"
        "conf" is the confidence score of the network.
        and is used by default as well as in the paper.
        Other metrics should be used with caution and/or
        experimentation.
        Other metrics:
            "iou" (intersection over union)
            "energy" (energy usage)
    min_connects : int, optional
        The minimum number of connections to use for the graph, by default 1
    cutoff : float, optional
        The cutoff to use for the graph, by default 0.0
    upper_outlier_cutoff : float, optional
        The upper outlier cutoff to use for the graph, by default 80.0
    connectivity : int, optional
        The connectivity to use for the graph, by default 2
    graph_type : Callable[[], nx.Graph | nx.DiGraph], optional
        The type of graph to use, by default None
        If None, will use nx.Graph.
    characterize_models : bool, optional
        Whether to characterize the models, by default None
        If None, will set to True and will characterize the
        models if the output directory does not contain a
        characterization.
    use_cached_energy : bool, optional
        Whether to use the cached energy values, by default None
        If None, will use the cached energy values if they exist.
    create_graph : bool, optional
        Whether or not to create the confidence graph.
        By default None, which will create the graph.
    purge_connectivity : bool, optional
        If True, the cutoff will be iteratively increased until the graph
        has the minimum number of connected components.
        By default None, which will not purge the graph.
    measure_load_stats : bool, optional
        Whether to measure the load time, energy, and memory usage of the model.
        By default None, which will measure the load time, energy, and memory usage.

    """
    if characterize_models is None:
        characterize_models = True
    if create_graph is None:
        create_graph = True
    if measure_load_stats is None:
        measure_load_stats = True

    if not Path.exists(output_dir):
        Path.mkdir(output_dir, parents=True)
        characterize_models = True

    jsonpath = output_dir / "steady_state.json"
    if not Path.exists(jsonpath):
        steady_state = get_steady_state(
            power_reader,
            get_memory,
            steady_state_sample_time,
        )
        with Path.open(jsonpath, "w") as f:
            json.dump(steady_state, f, ident=4)

    with Path.open(jsonpath, "r") as f:
        steady_state = json.load(f)
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
                image_dir=image_dir,
                image_names=image_files,
                ground_truth=ground_truth,
                num_power_iterations=num_power_iterations,
                dummy_image_size=dummy_image_size,
                use_cached=use_cached_energy,
                num_bins=num_bins,
            )
            del model

    if measure_load_stats:
        for model_func, model_name, import_func in zip(
            model_funcs,
            model_names,
            import_funcs,
        ):
            measure_load(
                output_dir=output_dir,
                model_func=model_func,
                modelname=model_name,
                power_reader=power_reader,
                memory_func=get_memory,
                import_handle=import_func,
                steady_state_sample_time=steady_state_sample_time,
            )

    if create_graph:
        build_graph(
            output_dir=output_dir,
            num_bins=num_bins,
            metric=graph_metric,
            min_connects=min_connects,
            cutoff=cutoff,
            upper_outlier_cutoff=upper_outlier_cutoff,
            connectivity=connectivity,
            graph_type=graph_type,
            purge_connectivity=purge_connectivity,
        )
