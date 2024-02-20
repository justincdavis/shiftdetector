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
Module for commonly used tools.

Functions
---------
change_pair
    Adjust x/y pairs to be a minimum size within some bounds.
sanitize_bbox
    Sanitize the bounding box to fit within the image dimensions and be a minimum size.
ncc
    Compute the normalized cross-correlation between two images.
"""
from __future__ import annotations

import contextlib

import cv2  # type: ignore[import-untyped]
import numpy as np


def clamp(n: float, minval: float, maxval: float) -> int | float:
    """
    Clamp a number between a minimum and maximum value.

    Parameters
    ----------
    n : float
        The number to clamp.
    minval : float
        The minimum value.
    maxval : float
        The maximum value.

    """
    if n < minval:
        return minval
    if n > maxval:
        return maxval
    return n


def bbox_iou(
    box1: tuple[int, int, int, int],
    box2: tuple[int, int, int, int],
    eps: float = 1e-7,
) -> float:
    """
    Compute IoU between two bounding boxes of x1, y1, x2, y2 format.

    Parameters
    ----------
    box1: tuple[int, int, int, int]
        The first bounding box (x1, y1, x2, y2)
    box2: tuple[int, int, int, int]
        The second bounding box (x1, y1, x2, y2)
    eps: float, optional
        A small value to prevent division by zero, by default 1e-7

    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter = max((min(b1_x2, b2_x2) - max(b1_x1, b2_x1)), 0) * max(
        (min(b1_y2, b2_y2) - max(b1_y1, b2_y1)),
        0,
    )

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    return inter / union


def compute_map(
    ground_truth: list[tuple[tuple[int, int, int, int], int]],
    model_output: list[tuple[tuple[int, int, int, int], int, int]],
    iou_threshold: float = 0.5,
) -> tuple[float, float, float]:
    """
    Compute the mAP between ground truth and model output.

    Parameters
    ----------
    ground_truth: list[tuple[tuple[int, int, int, int], int]]
        The ground truth bounding boxes and classes.
    model_output: list[tuple[tuple[int, int, int, int], int, int]]
        The model output bounding boxes, classes, and confidence scores.
    iou_threshold: float, optional
        The IoU threshold to use, by default 0.5

    Returns
    -------
    tuple[float, float]
        The mean average precision (mAP) between the ground truth and model output,
        and the average iou for detections.

    """
    average_precision = 0.0
    iou_vals = []
    conf_vals = []

    num_predictions = len(model_output)
    if num_predictions == 0:
        if len(ground_truth) == 0:
            return 1.0, 0.0, 0.0
        return 0.0, 0.0, 0.0

    sorted_predictions = sorted(model_output, key=lambda x: x[1], reverse=True)

    for gt_box in ground_truth:
        true_positives = 0

        for pred_box in sorted_predictions:
            iou = bbox_iou(gt_box[0], pred_box[0])
            if iou >= iou_threshold and gt_box[1] == pred_box[1]:
                true_positives += 1
                iou_vals.append(iou)
                conf_vals.append(pred_box[2])

        precision = true_positives / num_predictions

        average_precision += precision

    return (
        float(average_precision / len(ground_truth)),
        float(np.mean(iou_vals)),
        float(np.mean(conf_vals)),
    )


def scale_coords(
    bbox: tuple[int, int, int, int],
    s1: tuple[int, int],
    s2: tuple[int, int],
) -> tuple[int, int, int, int]:
    """
    Scale a bounding box from one image size to another.

    Function which takes a bounding box (x1, y1, x2, y2) within the image size
    s1 (width, height) and transform it to be a bounding box (x1, y1, x2, y2)
    for the image size s2 (width, height)

    Parameters
    ----------
    bbox: tuple[int, int, int, int]
        The bounding box (x1, y1, x2, y2) in image size s1 (width, height)
    s1: tuple[int, int]
        The input image size the bbox is based on
    s2: tuple[int, int]
        The output image size to transform the bbox to

    Returns
    -------
    tuple[int, int, int, int]
        A (x1, y1, x2, y2) bounding box in the s2 image size

    """
    x1, y1, x2, y2 = bbox  # Unpack the bounding box coordinates

    # Calculate scaling factors for x and y dimensions
    scale_x = s2[0] / s1[0]
    scale_y = s2[1] / s1[1]

    # Scale the coordinates using the calculated scaling factors
    scaled_x1 = int(x1 * scale_x)
    scaled_y1 = int(y1 * scale_y)
    scaled_x2 = int(x2 * scale_x)
    scaled_y2 = int(y2 * scale_y)

    return (scaled_x1, scaled_y1, scaled_x2, scaled_y2)


def change_pair(
    cords: tuple[int, int],
    maxval: int,
    minval: int,
    min_size: int = 10,
    max_iterations: int = 3,
) -> tuple[int, int]:
    """
    Adjust x/y pairs to be a minimum size within some bounds.

    Parameters
    ----------
    cords : tuple[int, int]
        The x/y pair to adjust.
    maxval : int
        The maximum value for the pair.
    minval : int
        The minimum value for the pair.
    min_size : int, optional
        The minimum size of the pair, by default 10.
    max_iterations : int, optional
        The maximum number of iterations to attempt to fix the pair, by default 3.

    Returns
    -------
    tuple[int, int]
        The adjusted x/y pair.

    """
    c1, c2 = cords
    counter = 0
    while counter < max_iterations:
        diff = c2 - c1
        if diff < min_size:
            offset = int((min_size - diff) / 2)
            if c1 > (minval + offset):
                c1 -= offset
            if c2 < (maxval - offset):
                c2 += offset
        else:
            break
        counter += 1
    return c1, c2


def sanitize_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    min_size: int = 10,
    max_iterations: int = 3,
) -> tuple[int, int, int, int]:
    """
    Sanitize the bounding box to fit within the image dimensions and be a minimum size.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        The bounding box to sanitize.
    width : int
        The width of the image.
    height : int
        The height of the image.
    min_size : int, optional
        The minimum size of the bounding box, by default 10.
    max_iterations : int, optional
        The maximum number of iterations to attempt to fix the bounding box, by default 3.

    Returns
    -------
    tuple[int, int, int, int]
        The sanitized bounding box.

    """
    x1, y1, x2, y2 = bbox
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, width)
    y2 = min(y2, width)
    x1, x2 = change_pair((x1, x2), width, 0, min_size, max_iterations)
    y1, y2 = change_pair((y1, y2), height, 0, min_size, max_iterations)
    return x1, y1, x2, y2


def ncc(
    image1: np.ndarray,
    image2: np.ndarray,
    size: tuple[int, int] | None = (112, 112),
) -> float:
    """
    Compute the normalized cross-correlation between two images.

    Parameters
    ----------
    image1 : np.ndarray
        The first image. Can be color or grayscale.
        Converted to grayscale if color.
    image2 : np.ndarray
        The second image. Can be color or grayscale.
        Converted to grayscale if color.
    size : tuple[int, int], optional
        The size to resize the images to, by default (112, 112).
        If None, the images are not resized.

    Returns
    -------
    float
        The normalized cross-correlation between the two images.

    """
    colorchannels = 3
    with contextlib.suppress(IndexError):
        if image1.shape[2] == colorchannels:  # type: ignore[misc]
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    with contextlib.suppress(IndexError):
        if image2.shape[2] == colorchannels:  # type: ignore[misc]
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    if size is not None:
        image1 = cv2.resize(image1, size, interpolation=cv2.INTER_LINEAR)
        image2 = cv2.resize(image2, size, interpolation=cv2.INTER_LINEAR)

    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    image1_numerator = image1 - np.mean(image1) / np.std(image1)
    image2_numerator = image2 - np.mean(image2) / np.std(image2)

    return float(
        np.sum(image1_numerator * image2_numerator)
        / (np.sqrt(np.sum(image1_numerator**2)) * np.sqrt(np.sum(image2_numerator**2))),
    )
