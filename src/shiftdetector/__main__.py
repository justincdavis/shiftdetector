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
"""Run command line utilities for the shift detector package."""
from __future__ import annotations

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the shift detector.")
    parser.add_argument("--run", required=False, action="store_true", help="Run a model on a video.")
    args = parser.parse_args()

    if args.run:
        from shiftdetector.cli import run_cli
        run_cli()
