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
Module for types used in implementation.

Classes
-------
SimValidation
    Enum for the simulation data return codes.
"""
from __future__ import annotations

from enum import Enum


class SimValidation(Enum):
    """
    Enum for the simulation validation type.

    Attributes
    ----------
    CORRECT : int
        The sim data is correct.
    NAMES : int
        Model names are incorrect type.
    DATA : int
        Model data is incorrect type.
    SIZES : int
        Data sizes do not match.
    BBOX_TYPE : int
        Bounding box is incorrect type.
    SCORE_TYPE : int
        Score is incorrect type.

    """

    CORRECT = 0
    NAMES = 1
    DATA = 2
    SIZES = 3
    BBOX_TYPE = 4
    SCORE_TYPE = 5
