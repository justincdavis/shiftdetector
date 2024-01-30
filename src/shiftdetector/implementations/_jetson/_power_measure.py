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

from threading import Condition, Thread

import jtop  # type: ignore[import-not-found]
from typing_extensions import Self

from shiftdetector.characterization import AbstractMeasure


class _ThreadJTOP:
    """Thread for JTOP."""

    def __init__(self: Self, interval: float) -> None:
        """Initialize the JTOP thread."""
        self._interval = interval
        self._stopped = False
        self._power = 0.0
        self._condition = Condition()
        self._thread = Thread(target=self._run)

    @property
    def power(self: Self) -> float:
        """Get the power."""
        return self._power

    def start(self: Self) -> None:
        """Start the JTOP thread."""
        with self._condition:
            self._condition.notify()

    def stop(self: Self) -> None:
        """Stop the JTOP thread."""
        self._stopped = True
        self._thread.join()

    def _run(self: Self) -> None:
        """Run the JTOP thread."""
        with jtop.jtop(self._interval) as jetson:
            while not self._stopped and jetson.ok():
                self._power = jetson.power["tot"]["power"]
        with jtop.jtop(1) as jetson:
            return


class JetsonPowerMeasure(AbstractMeasure):
    """Power measure for the Jetson series."""

    def __init__(self: Self, interval: float = 0.05) -> None:
        """Initialize the Jetson power measure."""
        self._jtop = _ThreadJTOP(interval)
        super().__init__()

    def start(self: Self) -> None:
        self._jtop.start()
        super().start()

    def stop(self: Self) -> None:
        super().stop()
        self._jtop.stop()

    def measure_data(self: Self) -> float:
        """Measure the power usage of the Jetson."""
        return self._jtop.power
