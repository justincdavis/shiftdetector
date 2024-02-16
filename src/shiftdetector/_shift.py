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

from typing import TYPE_CHECKING, Callable

from ._loader import AbstractModelLoader, DynamicModelLoader, SimulatedModelLoader
from ._scheduler import ShiftScheduler

if TYPE_CHECKING:
    from typing_extensions import Self

    from .characterization import AbstractModel


class Shift:
    """
    The SHIFT methodology as described in the DATE24 paper.

    Paper link: https://arxiv.org/abs/2402.07415
    """

    def __init__(
        self: Self,
        stats_dir: str,
        get_memory: Callable[[], tuple[float, float, float]],
        model_data: list[tuple[str, Callable[[], AbstractModel], float]],
        cost_threshold: float = 1.0,
        accuracy_threshold: float = 0.5,
        momentum: int = 10,
        solve_method: str = "greedy",
        knobs: dict[str, float] | None = None,
        sim_data: (
            dict[str, list[tuple[tuple[int, int, int, int], float]]] | None
        ) = None,
        *,
        simulated: bool | None = None,
    ) -> None:
        """
        Initialize the SHIFT methodology.

        Parameters
        ----------
        stats_dir : str
            The directory containing the model statistics.
            This is created through the model characterization process.
        get_memory: Callable[[], tuple[float, float, float]]
            A function that returns the total memory, available memory, and used memory.
        model_data: list[tuple[str, Callable[[], AbstractModel]]]
            A list of tuples of model name, function to create the model, and the cost to load the model into memory.
        cost_threshold : float, optional
            The cost threshold to use when determining which models are
            potential candidates for improving or maintaining the accuracy.
            The default is 0.5.
            The higher the value the more models are included for candidacy.
            The lower the value the less models are included for candidacy.
        accuracy_threshold : float, optional
            The accuracy to target when choosing which models to schedule.
            If the predicted accuracy is below this threshold, then the model
            will not be scheduled.
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
        sim_data : dict[str, list[tuple[tuple[int, int, int, int], float]]], optional
            The simulated data to use when running the SHIFT methodology in a
            simulated environment.
            The default is None.
            The sim_data dict (if provided) should contain model names as keys
            and a list of tuples of bounding boxes and scores as values.
        simulated : bool, optional
            Whether to run the SHIFT methodology in a simulated environment.
            The default is None. If None, then false is used.
            If True, then the model loading process will be simulated, and
            the model results will be loaded from files contained in the
            stats_dir, which were generated during the model characterization.

        Raises
        ------
        ValueError
            If the sim_data parameter is not provided when simulated is True.

        """
        self._scheduler = ShiftScheduler(
            stats_dir=stats_dir,
            cost_threshold=cost_threshold,
            accuracy_threshold=accuracy_threshold,
            momentum=momentum,
            solve_method=solve_method,
            knobs=knobs,
        )
        self._simulated = simulated if simulated is not None else False
        if self._simulated and sim_data is None:
            err_msg = "The sim_data parameter must be provided if simulated is True."
            raise ValueError(err_msg)
        self._sim_data = sim_data

        self._dml: AbstractModelLoader | None = None
        if not self._simulated:
            self._dml = DynamicModelLoader(
                get_memory=get_memory,
                model_data=model_data,
            )
        else:
            self._dml = SimulatedModelLoader(
                get_memory=get_memory,
                model_data=model_data,
            )
        if self._dml is None:
            err_msg = "The model loader has not been intialized. Internal error."
            raise ValueError(err_msg)
