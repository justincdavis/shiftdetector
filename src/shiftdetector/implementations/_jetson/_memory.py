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

import jtop  # type: ignore[import-not-found]


def get_jetson_memory() -> tuple[int, int, int]:
    with jtop.jtop(0.1) as jetson:
        while jetson.ok():
            tot_mem = jetson.memory["RAM"]["tot"]
            mem_free = jetson.memory["RAM"]["free"]
            mem_used = jetson.memory["RAM"]["used"]
    with jtop.jtop(1) as jetson:
        return tot_mem, mem_free, mem_used
