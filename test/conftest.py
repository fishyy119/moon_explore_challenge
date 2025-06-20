"""Path hack to make tests work."""

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parents[1]

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(TEST_DIR)  # to import this file from test code.
sys.path.append(ROOT_DIR.as_posix())
sys.path.append((ROOT_DIR / "moon_explore/hybrid_a_star").as_posix())


def run_this_test(file):
    pytest.main(args=["-W", "error", "-Werror", "--pythonwarnings=error", os.path.abspath(file)])
