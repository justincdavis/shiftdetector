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
# ruff: noqa: E402
"""
Package for the implementation for the SHIFT methodology.

This package contains the Python implementation of the SHIFT methodology. The
SHIFT methodology is a method for switching between object detection models
and accelerators based on a continuous data stream and ahead-of-time model
characterization. The SHIFT methodology is designed to be used with any
off-the-shelf object detection model available in Python.

Submodules
----------
characterization:
    Module for characterizing object detection models.
implementations:
    Module for implementations of measurement tools on hardware.
common:
    Module for common tools used in the SHIFT methodology.

Functions
---------
characterize:
    Characterize given object detection models.
"""
from __future__ import annotations

# setup the logger before importing anything else
import logging
import os
import sys


# Created from answer by Dennis at:
# https://stackoverflow.com/questions/7621897/python-logging-module-globally
def _setup_logger() -> None:
    # get logging level environment variable
    level = os.getenv("SHIFTDETECTOR_LOG_LEVEL")
    if level is not None:
        level = level.upper()
    level_map: dict[str | None, int] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        None: logging.WARNING,
    }
    log_level = level_map[level]

    # create logger
    logger = logging.getLogger(__package__)
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)


_setup_logger()
_log = logging.getLogger(__name__)


from . import characterization, common, implementations
from .characterization import characterize

__all__ = ["characterization", "characterize", "common", "implementations"]
__version__ = "0.0.0"

_log.info(f"Initialized shiftdetector with version {__version__}")
