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

from typing import TYPE_CHECKING

import cv2
import numpy as np
from trtutils import TRTEngine

from shiftdetector.characterization import AbstractModel
from shiftdetector.common import scale_coords

if TYPE_CHECKING:
    from typing_extensions import Self


class YoloModel(AbstractModel):
    def __init__(
        self: Self,
        engine_path: str,
        warmup_iterations: int = 1,
        dtype: np.number = np.float32,  # type: ignore[assignment]
        *,
        warmup: bool = False,
    ) -> None:
        """
        Create a YoloV7 model with TensorRT.

        Parameters
        ----------
        engine_path : str
            The path to the engine file.
        warmup_iterations : int
            The number of warmup iterations.
        dtype : np.number
            The data type of the model.
            Default is np.float32.
        warmup : bool
            Whether to run a warmup cycle.
            Default is False.

        """
        self._engine = TRTEngine(engine_path, warmup_iterations, dtype, warmup=warmup)
        self._dtype = dtype

    @property
    def input_size(self: Self) -> tuple[int, int]:
        """
        Get the input size of the model.

        Returns
        -------
        tuple[int, int]
            The input size of the model.

        """
        img_size = self._engine.input_shapes[0]
        return img_size[3], img_size[2]

    # Implementation provided by original YoloV7 authors
    # for use with TensorRT as per the yolob7trt.ipynb notebook
    @staticmethod
    def _letterbox(
        img: np.ndarray,
        new_shape: tuple[int, int] = (640, 640),
        color: tuple[int, int, int] = (114, 114, 114),
        stride: int = 32,
        *,
        auto: bool | None = None,
        scale_fill: bool | None = None,
        scaleup: bool | None = None,
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
        if auto is None:
            auto = True
        if scale_fill is None:
            scale_fill = False
        if scaleup is None:
            scaleup = True
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scale_fill:  # stretch
            dw, dh = 0, 0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

        dw_f = dw / 2  # divide padding into 2 sides
        dh_f = dh / 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh_f - 0.1)), int(round(dh_f + 0.1))
        left, right = int(round(dw_f - 0.1)), int(round(dw_f + 0.1))
        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=color,
        )  # add border
        return img, ratio, (dw_f, dh_f)

    def preprocess(self: Self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess the data for the model.

        Parameters
        ----------
        data : np.ndarray
            The data to preprocess.

        Returns
        -------
        np.ndarray
            The preprocessed data.

        """
        if data.shape[0] != self.input_size[1] or data.shape[1] != self.input_size[0]:
            data = cv2.resize(data, self.input_size, cv2.INTER_LINEAR)  # type: ignore[call-overload]
        data = self._letterbox(data)[0]
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = data / 255.0  # type: ignore[operator]
        data = data[np.newaxis, :]
        data = np.transpose(data, (0, 3, 1, 2))
        # ascontiguousarray has a large performance impact,
        # but it is needed for using cv2 operations later (if needed)
        return np.ascontiguousarray(data).astype(self._dtype)

    def predict(
        self: Self,
        data: np.ndarray,
    ) -> tuple[tuple[int, int, int, int], float]:
        """
        Predict on the input data.

        Parameters
        ----------
        data : np.ndarray
            The input data.

        Returns
        -------
        tuple[tuple[int, int, int, int], float]
            The output data, a bounding box and a score.

        """
        return self._execute(data, preprocessed=True)

    def _execute(
        self: Self,
        img: np.ndarray,
        *,
        preprocessed: bool | None = None,
    ) -> tuple[tuple[int, int, int, int], float]:
        if preprocessed is None:
            preprocessed = False
        if not preprocessed:
            img_height, img_width, _ = img.shape  # type: ignore[misc]
            img = self.preprocess(img)
        else:
            _, _, img_height, img_width = img.shape  # type: ignore[misc]
        _, _, new_height, new_width = img.shape  # type: ignore[misc]
        output = self._engine.execute([img])
        score = output[2][0][0]
        # print(score)
        x1, y1, x2, y2 = output[1][0][0]
        # print(x1, y1, x2, y2)
        x1, y1, x2, y2 = scale_coords(
            (x1, y1, x2, y2),
            (new_width, new_height),
            (img_width, img_height),
        )
        return (int(x1), int(y1), int(x2), int(y2)), score

    def warmup(self: Self) -> None:
        """Run a warmup cycle."""
        self._engine.mock_execute()

    def __call__(
        self: Self,
        img: np.ndarray,
        *,
        preprocessed: bool | None = None,
    ) -> tuple[tuple[int, int, int, int], float]:
        """
        Run the model on the input data.

        Parameters
        ----------
        img : np.ndarray
            The input data.
        preprocessed : bool | None
            Whether the input data is preprocessed.
            If None, the input data will be assumed to be
            not preprocessed.

        """
        return self._execute(img, preprocessed=preprocessed)
