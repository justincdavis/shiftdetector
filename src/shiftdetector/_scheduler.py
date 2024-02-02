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
import math
import os
import pickle  # noqa: S403
from collections import defaultdict, deque
from pathlib import Path

import networkx as nx
import numpy as np
from typing_extensions import Self


class Shift:
    """
    Class for the shift algorithm.

    This class is used to determine which models are potential
    candidates for improving or maintaining the accuracy of the
    model currently executing.
    """

    def __init__(
        self: Self,
        stats_dir: str,
        cost_threshold: float = 1.0,
        accuracy_threshold: float = 0.5,
        momentum: int = 10,
        solve_method: str = "greedy",
        knobs: dict[str, float] | None = None,
    ) -> None:
        """
        Use to intialize the shift algorithm.

        Parameters
        ----------
        stats_dir : str
            The directory containing the model statistics.
            This is created through the model characterization process.
        cost_threshold : float, optional
            The cost threshold to use when determining which models are
            potential candidates for improving or maintaining the accuracy.
            The default is 0.5.
            The higher the value the more models are included for candidacy.
            The lower the value the less models are included for candidacy.
        momentum : int, optional
            The number of previous accuracy estimates to use when determining
            the current accuracy of a model.
            The default is 10.
            The smaller the momentum the more reactive the algorithm is to
            changes in the accuracy.
            The larger the momentum the less reactive the algorithm is to
            changes in the accuracy.
        solve_method : str, optional
            The method to use when solving the shift algorithm.
            The default is "greedy".
            The other option is "optimal".
            The "greedy" method is faster but does not guarantee an optimal
            solution based on the heuristic.
            The "optimal" method is slower but does guarantee an optimal
            solution based on the heuristic.
        knobs : dict[str, float], optional
            The knobs to use when running the heuristic.
            The default is None.
            The knobs dict (if provided) should contain, accuracy, latency,
            and energy as keys. All values should be floats.
        """
        self._stats_dir = stats_dir
        self._cost_threshold = cost_threshold
        self._accuracy_threshold = accuracy_threshold
        self._momentum = momentum
        if solve_method not in ["greedy", "optimal"]:
            err_msg = f"Invalid solve method {solve_method}"
            raise ValueError(err_msg)
        self._solve_method = solve_method
        self._knobs = {
            "accuracy": 0.85,
            "latency": 0.10,
            "energy": 0.10,
        } if knobs is None else knobs
        knob_keys = set(self._knobs.keys())
        if knob_keys != {"accuracy", "latency", "energy"}:
            err_msg = f"Invalid knob keys {knob_keys}"
            raise ValueError(err_msg)
        self._conf_graph_path = None
        self._iou_graph_path = None
        self._model_stats = {}
        for root, dirs, files in os.walk(self._stats_dir):
            for file in files:
                if "conf_graph" in file:
                    self._conf_graph_path = Path(root) / file
                elif "iou_graph" in file:
                    self._iou_graph_path = Path(root) / file
            for directory in dirs:
                modelname = directory
                directorypath = Path(root) / directory
                with Path.open(Path(directorypath) / f"{modelname}.json") as f:
                    self._model_stats[modelname] = json.load(f)
        self._conf_graph = nx.read_weighted_edgelist(self._conf_graph_path, nodetype=str)
        self._iou_graph = nx.read_weighted_edgelist(self._iou_graph_path, nodetype=str)
        self._possible_models = self._get_possible_models()

        # at this point public methods are usable
        self._moments = {
            m: deque(maxlen=self._momentum) for m in self._possible_models
        }
        self._static_attrs = {
            "accuracy": {
                n: 0.0 for n in self._possible_models
            },
            "energy": {
                n: float(self._model_stats[n]["energy"]["mean"]) for n in self._possible_models
            },
            "latency": {
                n: float(self._model_stats[n]["time"]["mean"]) for n in self._possible_models
            },
        }

        # get the number of bins
        self._num_bins = float(self._model_stats[self._possible_models[0]]["bins"]["num_bins"])

        # check stats dir for shift_candidates.pkl
        self._cache_path = Path(self._stats_dir) / "shift_candidates.pkl"
        if not Path.exists(self._cache_path):
            self._candidates = self._pregenerate()  # huge runtime speedup
            with Path.open(self._cache_path, "wb") as f:
                pickle.dump(self._candidates, f, pickle.HIGHEST_PROTOCOL)
        else:
            with Path.open(self._cache_path, "rb") as f:
                self._candidates = pickle.load(f)  # noqa: S301
                try:
                    self._candidates = self._candidates[self._cost_threshold]
                except KeyError:
                    new_candidates = self._pregenerate()
                    self._candidates = new_candidates[self._cost_threshold]
                    with Path.open(self._cache_path, "wb") as f:
                        pickle.dump(new_candidates, f, pickle.HIGHEST_PROTOCOL)

        # precompute the energy and latency values (inverted for bigger is bigger metrics)
        self._energy = [(m, 1.0 - (e / max(self._static_attrs["energy"].values()))) for m, e in self._static_attrs["energy"].items()]
        self._latency = [(m, 1.0 - (l / max(self._static_attrs["latency"].values()))) for m, l in self._static_attrs["latency"].items()]

        # # test out transform to energy/latency to "spread out" the values
        # self._energy = self._transform(self._energy)
        # self._latency = self._transform(self._latency)

        # assign attributes for image similarity
        self._last_image: np.ndarray | None = None
        self._last_image_ncc: np.ndarray | None = None
        self._last_bbox_roi: np.ndarray | None = None
        self._last_bbox_ncc: np.ndarray | None = None

        # state tracking for last returned model
        self._last_model: str | None = None

    @staticmethod
    def _transform(data: list[tuple[str, float]]) -> list[tuple[str, float]]:
        models = [m for m, _ in data]
        vals = [e for _, e in data]
        new_vals = np.argsort(np.argsort(vals)).astype(float) / len(vals)
        return [(m, v) for m, v in zip(models, new_vals)]

    @property
    def possible_models(self) -> list[str]:
        """
        Use to get the possible models to use for inference.

        Returns
        -------
        list[str]
            The list of possible models to use for inference.
        """
        return self._possible_models

    @property
    def accuracy(self) -> float:
        """
        Use to get the accuracy knob value.

        Returns
        -------
        float
            The accuracy knob value.
        """
        return self._knobs["accuracy"]

    @accuracy.setter
    def accuracy(self, value: float) -> None:
        """
        Use to set the accuracy knob value.

        Parameters
        ----------
        value : float
            The accuracy knob value to use.
        """
        self._knobs["accuracy"] = value

    @property
    def latency(self) -> float:
        """
        Use to get the latency knob value.

        Returns
        -------
        float
            The latency knob value.
        """
        return self._knobs["latency"]

    @latency.setter
    def latency(self, value: float) -> None:
        """
        Use to set the latency knob value.

        Parameters
        ----------
        value : float
            The latency knob value to use.
        """
        self._knobs["latency"] = value

    @property
    def energy(self) -> float:
        """
        Use to get the energy knob value.

        Returns
        -------
        float
            The energy knob value.
        """
        return self._knobs["energy"]

    @energy.setter
    def energy(self, value: float) -> None:
        """
        Use to set the energy knob value.

        Parameters
        ----------
        value : float
            The energy knob value to use.
        """
        self._knobs["energy"] = value

    @property
    def knobs(self) -> tuple[float, float, float]:
        """
        Use to get the knobs.

        Returns
        -------
        tuple[float, float, float]
            The knobs.
        """
        return self._knobs["accuracy"], self._knobs["latency"], self._knobs["energy"]

    @knobs.setter
    def knobs(
        self,
        values: tuple[float, float, float],
    ) -> None:
        """
        Use to set the knobs.

        Parameters
        ----------
        values : tuple[float, float, float]
            The knobs to use.
        """
        self._knobs["accuracy"], self._knobs["latency"], self._knobs["energy"] = values

    @staticmethod
    def _sanitize_bbox(bbox: tuple[int, int, int, int], width: int, height: int, min_size: int = 10) -> tuple[int, int, int, int]:
        def change_pair(cords: tuple[int, int], maxval: int, minval: int, min_size: int = 10) -> tuple[int, int]:
            c1, c2 = cords
            counter = 0
            while counter < 3:
                diff = c2 - c1
                if diff < min_size:
                    offset = int((min_size - diff) / 2)
                    if c1 > (minval + offset):
                        c1 -= offset
                    if c2 < (maxval - offset):
                        c2 += offset
                else:
                    break
                counter += 1
            return c1, c2
        x1, y1, x2, y2 = bbox
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, width)
        y2 = min(y2, width)
        x1, x2 = change_pair((x1, x2), width, 0, min_size)
        y1, y2 = change_pair((y1, y2), height, 0, min_size)
        return x1, y1, x2, y2

    def _ncc(self, image: np.ndarray, bbox_roi: tuple[int, int, int, int]) -> float:
        bbox_roi = self._sanitize_bbox(bbox_roi, image.shape[1], image.shape[0])
        bbox_ncc = self._ncc_bbox(image[bbox_roi[1]:bbox_roi[3], bbox_roi[0]:bbox_roi[2]])
        image_ncc = self._ncc_image(image)
        return max(bbox_ncc * image_ncc, 0.0)

    def _pregenerate(self) -> None:
        accuracy_estimates = {}
        for modelname in self.get_possible_models():
            for i in range(1, 100, 1):
                confidence = i / 100
                # get the node name
                node_name = self._get_node_name(modelname, confidence)
                # get the neighbors of the node
                estimates = self._get_accuracy_estimates(node_name)
                # save to dict
                accuracy_estimates[node_name] = estimates
        # check if the cache file already exists
        cache_data = {}
        if os.path.exists(self._cache_path):
            with open(self._cache_path, "rb") as f:
                cache_data = pickle.load(f)  # the cache data is a dictionary
        cache_data[self._cost_threshold] = accuracy_estimates
        return cache_data

    def _get_possible_models(self) -> list[str]:
        nodes = list(self._conf_graph.nodes())
        nodes = {n[:n.rfind("_")] for n in nodes}
        # extra iteration to fold off processing unit (but should keep)
        # nodes = set([n[:n.rfind("_")] if "_" in n else n for n in nodes])
        return sorted(list(nodes))

    def _conf_to_bin(self, confidence: float) -> float:
        confidence = math.ceil(confidence * self._num_bins) / self._num_bins
        return confidence

    def _get_node_name(self, modelname: str, raw_confidence: float) -> str:
        confidence = self._conf_to_bin(raw_confidence)
        return f"{modelname}_{confidence}"

    def _get_neighbors(self, node_name: str) -> list:
        try:
            edges = nx.bfs_tree(self._conf_graph, node_name).edges()
            weights = [self._conf_graph.get_edge_data(*e)["weight"] for e in edges]
            edges = [(e, w) for e, w in zip(edges, weights) if w <= self._cost_threshold]
            nodes = [(e[1], w) for e, w in edges]
            return nodes
        except nx.exception.NetworkXError:
            return []

    def _get_accuracy_estimate(self, node_name: str, raw_confidence: float | None = None) -> tuple[str, float]:
        idx = node_name.rfind("_")
        modelname = node_name[:idx]
        confidence = node_name[idx + 1:]
        if raw_confidence is None:
            accuracy = float(self._model_stats[modelname]["bins"][confidence]["iou_mean"])
        else:
            fit_data = self._model_stats[modelname]["bins"][confidence]["fit"]
            fit_data = [f for f in fit_data.split(",")]
            slope = float(fit_data[0].replace("[", ""))
            intercept = float(fit_data[1].replace("]", ""))
            accuracy = slope * raw_confidence + intercept
        return node_name, accuracy

    def _get_accuracy_estimates(self, node_name: str) -> list:
        current_accuracy_estimate = self._get_accuracy_estimate(node_name)
        neighbors = self._get_neighbors(node_name)
        # get the accuracy estimate of the neighbors and current
        accuracy_estimates = [(*self._get_accuracy_estimate(n), w) for n, w in neighbors]
        accuracy_estimates.append((*current_accuracy_estimate, 0.0))  # current model has 0 cost
        # fold models with the same name into an averaged estimate
        modelname_to_accuracy = defaultdict(lambda: (list(), list()))
        for _ in range(1):  # do not fold processing units since energy/latency is not the same
        # for _ in range(2):  # two iterations, one to fold off confidence, one to fold processors
            temp_m2a = defaultdict(lambda: (list(), list()))
            for n, a, w in accuracy_estimates:
                n_idx = len(n) + 1
                if "_" in n:
                    n_idx = n.rfind("_")
                if isinstance(a, list):
                    temp_m2a[n[:n_idx]][0].extend(a)
                    temp_m2a[n[:n_idx]][1].extend(w)
                else:
                    temp_m2a[n[:n_idx]][0].append(a)
                    temp_m2a[n[:n_idx]][1].append(w)
            modelname_to_accuracy = temp_m2a
            accuracy_estimates = [
                (n, a, w) for n, (a, w) in modelname_to_accuracy.items()
            ]
        accuracy_estimates = []
        for mname, (accuracies, weights) in modelname_to_accuracy.items():
            inv_weights = [max(((self._cost_threshold - w) / self._cost_threshold) ** 2, 1e-8) for w in weights]
            est_acc = np.average(accuracies, weights=inv_weights)
            harmonic_mean_weight = len(weights) / sum([1.0 / max(w, 1e-8) for w in weights])
            accuracy_estimates.append((mname, est_acc, harmonic_mean_weight))

        return accuracy_estimates

    def _get_candidates(self, modelname: str, confidence: float) -> str:
        # get the node name
        confidence = self._conf_to_bin(confidence)
        node_name = self._get_node_name(modelname, confidence)
        # print(f"Model: {modelname}, Confidence: {confidence}, Node: {node_name}")
        # get the neighbors of the node
        try:
            return self._candidates[node_name]
        except Exception:
            return self._get_accuracy_estimates(node_name)

    def __call__(self, modelname: str, confidence: float, image: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
        # solve the ncc
        ncc = self._ncc(image, bbox)
        if ncc * confidence >= self._accuracy_threshold and self._last_model is not None:
            return self._last_model

        # get info for all models again
        var_attrs = self._static_attrs.copy()

        # get candidates based on conf_graph
        candidates = self._get_candidates(modelname, confidence)

        # fill out var_attrs with the candidates info
        for c in candidates:
            # acc_adjust = (self._cost_threshold - (c[2] * math.sqrt(ncc))) / self._cost_threshold
            # acc_adjust = (self._cost_threshold - c[2]) / self._cost_threshold
            acc_adjust = 1.0
            ajd_acc = c[1] * acc_adjust

            # print(f"Model: {c[0]}, Accuracy: {c[1]}, Cost: {c[2]}, Adjusted Accuracy: {ajd_acc}")
            var_attrs["accuracy"][c[0]] = ajd_acc
            # var_attrs["accuracy"][c[0]] = c[1]
            self._moments[c[0]].append(var_attrs["accuracy"][c[0]])
            var_attrs["accuracy"][c[0]] = np.mean(self._moments[c[0]])

        if self._solve_method == "greedy":
            # scale energy and latency to be 0 to 1 metrics
            # also reverse the values so that higher is better
            # leave accuracy alone since already 0 to 1 metrics
            accuracy = [(m, a) for m, a in var_attrs["accuracy"].items()]

            model_data = []
            for (m, a), (_, e), (_, l) in zip(accuracy, self._energy, self._latency):
                if a >= self._accuracy_threshold:
                    model_data.append((m, a, e, l))
            if len(model_data) == 0:
                model_data = [(m, a, e, l) for (m, a), (_, e), (_, l) in zip(accuracy, self._energy, self._latency)]

            # # use the knobs to determine the fitness of each model
            # fitness = [
            #     (m, self._knobs["accuracy"] * a + self._knobs["energy"] * e + self._knobs["latency"] * l)
            #     for (m, a), (_, e), (_, l) in zip(accuracy, energy, latency)
            # ]
            fitness = [
                (m, a * self._knobs["accuracy"] + e * self._knobs["energy"] + l * self._knobs["latency"]) for m, a, e, l in model_data
            ]

            # for a, e, l in zip(accuracy, energy, latency):
            #     print(f"{a[0]}: {a[1]}, {e[1]}, {l[1]}")

            # sort the fitnesses and return the best model
            fitness = sorted(fitness, key=lambda x: x[1], reverse=True)
            best_model = fitness[0][0]
        else:
            raise NotImplementedError("Optimal solve method not implemented")

        self._last_model = best_model
        return best_model

    def get_possible_models(self) -> list[str]:
        """
        Use to get the possible models to use for inference.

        Returns
        -------
        list[str]
            The list of possible models to use for inference.
        """
        return self._possible_models