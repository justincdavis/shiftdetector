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

import argparse
from pathlib import Path
from typing import Callable

from shiftdetector.characterization import AbstractModel
from shiftdetector.cli import run


def _get_oakd_model(blobpath: str, input_size: tuple[int, int]) -> Callable[[], AbstractModel]:
    from shiftdetector.implementations.oakd import OakModel
    def _model() -> AbstractModel:
        return OakModel(blobpath, input_size=input_size)
    return _model

def _get_yolo_tensorrt_model(modelpath: str) -> Callable[[], AbstractModel]:
    from shiftdetector.implementations.tensorrt import YoloModel
    def _model() -> AbstractModel:
        return YoloModel(modelpath)
    return _model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a model on a video.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model.")
    parser.add_argument("--name", type=str, required=False, help="Name of the model.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--image_size", type=tuple[int, int], required=False, help="Size of the input image.")
    parser.add_argument("--platform", type=str, required=False, help="Platform to run the model on.")
    parser.add_argument("--powerdraw", type=float, required=False, help="Power draw of the model.")
    parser.add_argument("--char_dir", type=str, required=False, help="Path to the characterization directory.")
    args = parser.parse_args()

    modelname = args.name if args.name else Path(args.model).stem

    # handle modelfunc
    extension = Path(args.model).suffix
    modelfunc = None
    if extension == ".blob":
        if args.input_size is None:
            err_msg = "Input size must be provided for OAK-D models."
            raise ValueError(err_msg)
        modelfunc = _get_oakd_model(args.model, args.input_size)
    elif extension == ".engine" or extension == ".trt":
        if "yolo" in modelname.lower():
            modelfunc = _get_yolo_tensorrt_model(args.model)
        else:
            err_msg = f"Do not have an implementation available for model {modelname},"
            err_msg = " using TensorRT. Currently only YoloV7 is supported."
            raise ValueError(err_msg)

    # handle power_measure
    power_measure = None
    if args.platform is not None:
        if args.platform == "jetson":
            from shiftdetector.characterization import JetsonMeasure
            power_measure = JetsonMeasure()
        else:
            err_msg = f"Do not have an implementation available for platform {args.platform}."
            raise ValueError(err_msg)

    if modelfunc is None:
        err_msg = f"Do not have an implementation available for model {modelname}."
        raise ValueError(err_msg)

    run(
        videofile=args.video,
        output_dir=args.output_dir,
        modelname=args.model,
        modelfunc=modelfunc,
        power_measure=power_measure,
        power_draw=args.powerdraw,
        characterization_dir=args.char_dir,
    )
