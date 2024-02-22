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
# ruff: noqa: N801
"""Module for implemenations on hardware."""
from __future__ import annotations

from typing_extensions import Self

try:
    from . import _jetson as jetson
except ImportError:

    class jetson:  # type: ignore[no-redef]
        """Mock class for Jetson implementation."""

        def __getattr__(self: Self, name: str) -> None:
            """Error message for Jetson."""
            err_msg = "Jetson implementation not available."
            err_msg += " Please install jetson dependencies."
            raise ImportError(err_msg)


try:
    from . import _oakd as oakd
except ImportError:

    class oakd:  # type: ignore[no-redef]
        """Mock class for Oak-D implementation."""

        def __getattr__(self: Self, name: str) -> None:
            """Error message for Oak-D."""
            err_msg = "Oak-D implementation not available."
            err_msg += " Please install oakd dependencies."
            raise ImportError(err_msg)


from . import _mock as mock

__all__ = ["jetson", "mock", "oakd"]
