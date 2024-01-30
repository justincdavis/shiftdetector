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
"""
from __future__ import annotations

from ._method import characterize
from ._types import AbstractMeasure

__all__ = ["AbstractMeasure", "characterize"]
