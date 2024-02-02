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
"""
Module which contains tools for characterization of object detection models.

This module contains tools for characterizing object detection models. The
characterization tools are used to generate a characterization of the model's
performance and resource usage.

Classes
-------
AbstractMeasure
    An abstract class for defining the interface for data measurement.
AbstractModel
    An abstract class for defining the interface for object detection models.

Functions
---------
build_graph
    Build a graph from the edges and weights.
characterize
    Characterize a list of object detection models.
get_power_draw
    Get the power draw of the system.
get_steady_state
    Get the steady state power and memory usage of the system.
measure_load
    Measure the load time, energy, and memory usage of a model.
"""
from __future__ import annotations

from ._graph import build_graph
from ._helpers import get_power_draw, get_steady_state
from ._load import measure_load
from ._method import characterize
from ._types import AbstractMeasure, AbstractModel

__all__ = [
    "AbstractMeasure",
    "AbstractModel",
    "build_graph",
    "characterize",
    "get_power_draw",
    "get_steady_state",
    "measure_load",
]
