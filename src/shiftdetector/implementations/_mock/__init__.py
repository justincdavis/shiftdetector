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
Module for mock implementations of hardware.

Classes
-------
MockPowerMeasure
    Class for measuring power on a mock device.

Functions
---------
get_mock_memory
    Get the memory usage on a mock device.
"""
from __future__ import annotations

from ._memory import get_mock_memory
from ._power_measure import MockPowerMeasure

__all__ = ["MockPowerMeasure", "get_mock_memory"]
