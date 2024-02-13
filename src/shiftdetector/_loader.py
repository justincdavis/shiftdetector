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

from threading import Thread
from typing import TYPE_CHECKING, Callable

from typing_extensions import Self

if TYPE_CHECKING:
    from .characterization import AbstractModel


class _ModelLoaderThread:
    """
    Manage the loading of the model.
    
    Attributes
    ----------
    model: type[AbstractModel] | None
        The model
    
    Methods
    -------
    load()
        Load the model
    unload()
        Unload the model
    """

    def __init__(
        self: Self,
        modelname: str,
        model_creation_func: Callable[[], type[AbstractModel]],
    ) -> None:
        """Create the model loader thread."""
        self._modelname = modelname
        self._model_creation_func = model_creation_func
        self._model: type[AbstractModel] | None = None
        self._thread: Thread | None = None

    @property
    def model(self: Self) -> type[AbstractModel] | None:
        """
        Get the model.

        Returns
        -------
        type[AbstractModel] | None
            The model
        """
        return self._model

    def load(self: Self) -> None:
        """Load the model."""
        if self._model is None and self._thread is None:
            self._thread = Thread(target=self._load_model)
            self._thread.start()

    def unload(self: Self) -> None:
        """Unload the model."""
        del self._model
        del self._thread
        self._model = None
        self._thread = None

    def _load_model(self: Self) -> None:
        """Load the model."""
        self._model = self._model_creation_func()
