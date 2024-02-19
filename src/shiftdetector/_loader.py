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

import contextlib
from abc import ABC, abstractmethod
from threading import Thread, Condition
from typing import TYPE_CHECKING, Callable

from typing_extensions import Self

from .characterization import DummyModel

if TYPE_CHECKING:
    from .characterization import AbstractModel


class AbstractModelLoader(ABC):
    """
    Abstract model loader.

    Methods
    -------
    get_available_models()
        Get the available models
    request(modelname)
        Request a model to be loaded into memory
    get_model(modelname)
        Get the model

    """

    @abstractmethod
    def __init__(
        self: Self,
        get_memory: Callable[[], tuple[float, float, float]],
        model_data: list[tuple[str, Callable[[], AbstractModel], float]],
    ) -> None:
        """Create the abstract model loader."""
        err_msg = "The __init__ method must be implemented."
        raise NotImplementedError(err_msg)

    @abstractmethod
    def get_available_models(self: Self) -> list[str]:
        """
        Get the available models.

        Returns
        -------
        list[str]
            The available models

        """
        err_msg = "The get_available_models method must be implemented."
        raise NotImplementedError(err_msg)

    @abstractmethod
    def request(self: Self, modelname: str, *, wait: bool | None = None) -> bool:
        """
        Request a model to be loaded into memory.

        Parameters
        ----------
        modelname: str
            The name of the model to request.
        wait: bool, optional
            If True, wait for the model to be loaded.

        Returns
        -------
        bool
            True if the model is present, False otherwise.
            If False, the model will start to be loaded.

        """
        err_msg = "The request method must be implemented."
        raise NotImplementedError(err_msg)

    @abstractmethod
    def get_model(self: Self, modelname: str) -> AbstractModel | None:
        """
        Get the model.

        Returns
        -------
        type[AbstractModel] | None
            The model

        """
        err_msg = "The get_model method must be implemented."
        raise NotImplementedError(err_msg)


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
        model_creation_func: Callable[[], AbstractModel],
    ) -> None:
        """Create the model loader thread."""
        self._modelname = modelname
        self._model_creation_func = model_creation_func
        self._model: AbstractModel | None = None
        self._thread: Thread | None = None
        self._condition: Condition | None = None
        self._present = False

    def __del__(self: Self) -> None:
        """Unload the model."""
        self.unload()

    @property
    def model(self: Self) -> AbstractModel | None:
        """
        Get the model.

        Returns
        -------
        type[AbstractModel] | None
            The model

        """
        return self._model

    @property
    def present(self: Self) -> bool:
        """
        Check if the model is present.

        Returns
        -------
        bool
            True if the model is present, False otherwise.

        """
        return self._present
    
    @property
    def loading(self: Self) -> bool:
        """
        Check if the model is loading.

        Returns
        -------
        bool
            True if the model is loading, False otherwise.

        """
        return self._thread is not None and self._thread.is_alive()

    def load(self: Self) -> None:
        """Load the model."""
        if self._model is None and self._thread is None:
            self._thread = Thread(target=self._load_model)
            self._condition = Condition()
            self._thread.start()
    
    def wait(self: Self) -> None:
        """Wait for the model to be loaded."""
        if self._thread is not None and self._thread.is_alive() and not self._present:
            with self._condition:
                self._condition.wait()

    def unload(self: Self) -> None:
        """Unload the model."""
        self._present = False
        if self._thread is not None:
            with contextlib.suppress(RuntimeError):
                self._thread.join()
            del self._thread
            self._thread = None
        del self._model
        self._model = None

    def _load_model(self: Self) -> None:
        """Load the model."""
        self._model = self._model_creation_func()
        self._present = True
        with self._condition:
            self._condition.notify_all()


class DynamicModelLoader(AbstractModelLoader):
    """
    Manage the loading of models based on memory constraints.

    Methods
    -------
    get_available_models()
        Get the available models
    request(modelname)
        Request a model to be loaded into memory
    get_model(modelname)
        Get the model

    """

    def __init__(
        self: Self,
        get_memory: Callable[[], tuple[float, float, float]],
        model_data: list[tuple[str, Callable[[], AbstractModel], float]],
    ) -> None:
        """
        Create the dynamic model loader.

        Parameters
        ----------
        get_memory: Callable[[], tuple[float, float, float]]
            A function that returns the total memory, available memory, and used memory.
        model_data: list[tuple[str, Callable[[], AbstractModel]]]
            A list of tuples of model name, function to create the model, and the cost to load the model into memory.

        """
        self._get_memory = get_memory
        self._model_loaders: dict[str, _ModelLoaderThread] = {
            modelname: _ModelLoaderThread(modelname, model_creation_func)
            for modelname, model_creation_func, _ in model_data
        }
        self._model_costs: dict[str, float] = {
            modelname: cost for modelname, _, cost in model_data
        }
        self._counter = 0
        self._model_counters: dict[str, int] = dict.fromkeys(self._model_loaders, 0)

    def __del__(self: Self) -> None:
        """Unload all models."""
        for model_loader in self._model_loaders.values():
            del model_loader

    def get_available_models(self: Self) -> list[str]:
        """
        Get the available models.

        Returns
        -------
        list[str]
            The available models

        """
        return [
            modelname
            for modelname, model_loader in self._model_loaders.items()
            if model_loader.present
        ]

    def request(self: Self, modelname: str, *, wait: bool | None = None) -> bool:
        """
        Request a model to be loaded into memory.

        Parameters
        ----------
        modelname: str
            The name of the model to request.

        Returns
        -------
        bool
            True if the model is present, False otherwise.
            If False, the model will start to be loaded.
        wait: bool, optional
            If True, wait for the model to be loaded.

        Raises
        ------
        ValueError
            If the modelname is was not submitted with a model creation function.
        RuntimeError
            If no models are present, but memory is still required.
            If the model attempting to be loaded exceeds system memory this can occur.

        """
        if modelname not in self._model_loaders:
            err_msg = (
                f"Model {modelname} was not submitted with a model creation function."
            )
            raise ValueError(err_msg)

        self._counter += 1
        self._model_counters[modelname] = self._counter

        is_present = self._model_loaders[modelname].present
        if is_present:
            return True

        _, available_memory, _ = self._get_memory()
        if available_memory < self._model_costs[modelname]:
            memory_required = self._model_costs[modelname] - available_memory
            recouped_memory = 0.0
            while recouped_memory < memory_required:
                oldest_model: str | None = None
                for modelname, counter in self._model_counters.items():
                    if (
                        oldest_model is None
                        or counter < self._model_counters[oldest_model]
                    ) and self._model_loaders[modelname].present:
                        oldest_model = modelname
                if oldest_model is None:
                    # right now raise a runtime error, BUT
                    # could just return False and not load the model if this occurs
                    err_msg = "No models are present, but memory is still required."
                    err_msg += " If the model attempting to be loaded exceeds system memory this can occur."
                    raise RuntimeError(err_msg)
                recouped_memory += self._model_costs[oldest_model]
                self._model_loaders[oldest_model].unload()

        if not self._model_loaders[modelname].loading:
            self._model_loaders[modelname].load()

            if wait:
                self._model_loaders[modelname].wait()
        return False

    def get_model(self: Self, modelname: str) -> AbstractModel | None:
        """
        Get the model.

        Returns
        -------
        type[AbstractModel] | None
            The model

        """
        return self._model_loaders[modelname].model


class SimulatedModelLoader(AbstractModelLoader):
    """
    Simulated model loader for faster generation of results.

    Methods
    -------
    get_available_models()
        Get the available models
    request(modelname)
        Request a model to be loaded into memory
    get_model(modelname)
        Get the model

    """

    def __init__(
        self: Self,
        get_memory: Callable[[], tuple[float, float, float]],
        model_data: list[tuple[str, Callable[[], AbstractModel], float]],
    ) -> None:
        """
        Create the simulated model loader.

        Parameters
        ----------
        get_memory: Callable[[], tuple[float, float, float]]
            A function that returns the total memory, available memory, and used memory.
            Only used at the beginning to get the total memory.
        model_data: list[tuple[str, Callable[[], AbstractModel]]]
            A list of tuples of model name, function to create the model, and the cost to load the model into memory.
            Same type for static type analysis and parity, but the loading functions are not used.

        """
        self._get_memory = get_memory
        self._model_loaders: dict[str, AbstractModel] = {
            modelname: DummyModel() for modelname, _, _ in model_data
        }
        self._model_costs: dict[str, float] = {
            modelname: cost for modelname, _, cost in model_data
        }
        self._counter = 0
        self._model_counters: dict[str, int] = dict.fromkeys(self._model_loaders, 0)

    def get_available_models(self: Self) -> list[str]:
        """
        Get the available models.

        Returns
        -------
        list[str]
            The available models

        """
        return [
            modelname for modelname, present in self._model_loaders.items() if present
        ]

    def request(self: Self, modelname: str) -> bool:
        """
        Request a model to be loaded into memory.

        Parameters
        ----------
        modelname: str
            The name of the model to request.

        Returns
        -------
        bool
            True if the model is present, False otherwise.
            If False, the model will start to be loaded.

        """
        self._counter += 1
        self._model_counters[modelname] = self._counter

        return True

    def get_model(self: Self, modelname: str) -> AbstractModel | None:
        """
        Get the model.

        Returns
        -------
        type[AbstractModel] | None
            The model

        """
        return self._model_loaders[modelname]
