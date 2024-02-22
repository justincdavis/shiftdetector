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

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from oakutils.vpu import VPU, YolomodelData

from shiftdetector.characterization import AbstractModel
from shiftdetector.common import scale_coords

if TYPE_CHECKING:
    from typing_extensions import Self


class OakModel(AbstractModel):
    def __init__(
        self: Self,
        modeldir: Path | str,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """
        Initialize the Oak model.

        Parameters
        ----------
        modeldir : Path | str
            The path to the directory containing the blob and json file.
            If the model is a Yolo exported model, the json file
            must be in the same directory with the same name.
            If the path has 'yolo' in the name, the model is
            assumed to be a Yolo model.
        input_size : tuple[int, int] | None
            The input size of the model.
            If None, the input size will be determined from the
            json file if a Yolo model.

        Raises
        ------
        FileNotFoundError
            If the model directory does not exist.
        NotADirectoryError
            If the model directory is not a directory.

        """
        super().__init__()
        modeldir = Path(modeldir)
        if not Path.exists(modeldir):
            err_msg = "The model directory does not exist."
            raise FileNotFoundError(err_msg)
        if not Path.is_dir(modeldir):
            err_msg = "The model directory is not a directory."
            raise NotADirectoryError(err_msg)
        contents = os.listdir(modeldir)
        self._modelpath = None
        self._jsonpath = None
        for file in contents:
            if file.endswith(".blob"):
                self._modelpath = Path(modeldir) / file
            if file.endswith(".json"):
                self._jsonpath = Path(modeldir) / file
        if self._modelpath is None:
            err_msg = "The model directory does not contain a .blob file."
            raise FileNotFoundError(err_msg)
        self._input_size = input_size
        self._vpu = VPU()
        if "yolo" in self._modelpath.name:
            if self._jsonpath is None:
                err_msg = "The model directory does not contain a .json file."
                raise FileNotFoundError(err_msg)
            jsonpath = self._modelpath.with_suffix(".json")
            with Path.open(jsonpath, "r") as f:
                jsondata = json.load(f)  # type: ignore[arg-type]
            nndata = jsondata["nn_config"]["NN_specific_metadata"]
            inputstr: str = jsondata["nn_config"]["input_size"]
            inputdim = int(inputstr.split("x")[0])
            self._input_size = (inputdim, inputdim)
            yolodata = YolomodelData(
                confidence_threshold=nndata["confidence_threshold"],
                iou_threshold=nndata["iou_threshold"],
                num_classes=nndata["classes"],
                coordinate_size=nndata["coordinates"],
                anchors=nndata["anchors"],
                anchor_masks=nndata["anchor_masks"],
            )
            self._vpu.reconfigure(
                blob_path=self._modelpath,
                yolo_data=yolodata,
                is_yolo_model=True,
            )
        else:
            self._vpu.reconfigure(blob_path=self._modelpath)

        if self._input_size is None:
            err_msg = "The input size of the model is not known."
            err_msg += " Please provide the input size."
            raise ValueError(err_msg)

    @property
    def input_size(self: Self) -> tuple[int, int]:
        """
        Get the input size of the model.

        Returns
        -------
        tuple[int, int]
            The input size of the model.

        """
        if self._input_size is None:
            err_msg = "The input size is None after initialization."
            raise RuntimeError(err_msg)
        return self._input_size

    @staticmethod
    def _frame_norm(
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> tuple[int, int, int, int]:
        normvals = np.full(len(bbox), frame.shape[0])
        normvals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normvals).astype(int)  # type: ignore[no-any-return]

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
        if data.shape[0] != self.input_size[0] or data.shape[1] != self.input_size[0]:
            data = cv2.resize(data, self.input_size, cv2.INTER_LINEAR)  # type: ignore[call-overload]
        return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    def _execute(
        self: Self,
        image: np.ndarray,
        *,
        preprocessed: bool | None = None,
    ) -> tuple[tuple[int, int, int, int], float]:
        """
        Execute the model on the input data.

        Parameters
        ----------
        image : np.ndarray
            The input data.
        preprocessed : bool | None
            Whether the input data is preprocessed.
            If None, the input data will be assumed to be
            preprocessed.

        Returns
        -------
        tuple[tuple[int, int, int, int], float]
            The output data, a bounding box and a score.

        """
        if not preprocessed:
            img_height, img_width, _ = image.shape  # type: ignore[misc]
            image = self.preprocess(image)
        else:
            img_height, img_width, _ = image.shape  # type: ignore[misc]
        new_height, new_width, _ = image.shape  # type: ignore[misc]
        output = self._vpu.run(image)
        if isinstance(output, np.ndarray):
            err_msg = "The model did not return an ImgDetection type."
            raise TypeError(err_msg)
        output_dects = output.detections
        best_bbox, max_conf = (0, 0, 0, 0), 0.0
        for d in output_dects:
            bbox = self._frame_norm(image, (d.xmin, d.ymin, d.xmax, d.ymax))
            confidence = d.confidence
            if confidence > max_conf:
                best_bbox = bbox
                max_conf = confidence
        x1, y1, x2, y2 = scale_coords(
            best_bbox,
            (new_width, new_height),
            (img_width, img_height),
        )
        return (int(x1), int(y1), int(x2), int(y2)), max_conf

    def predict(
        self: Self,
        data: np.ndarray,
    ) -> tuple[tuple[int, int, int, int], float]:
        """
        Predict the output data from the input data.

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

    def __call__(
        self: Self,
        data: np.ndarray,
        *,
        preprocessed: bool | None = None,
    ) -> tuple[tuple[int, int, int, int], float]:
        """
        Predict the output data from the input data.

        Parameters
        ----------
        data : np.ndarray
            The input data.
        preprocessed : bool | None
            Whether the input data is preprocessed.
            If None, the input data will be assumed to be
            preprocessed.

        Returns
        -------
        tuple[tuple[int, int, int, int], float]
            The output data, a bounding box and a score.

        """
        return self._execute(data, preprocessed=preprocessed)
