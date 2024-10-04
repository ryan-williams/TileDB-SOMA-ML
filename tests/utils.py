# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

from functools import partial
from typing import Callable, Union

import numpy as np
from scipy.sparse import coo_matrix, spmatrix

from tiledbsoma_ml.pytorch import (
    ExperimentAxisQueryIterable,
    ExperimentAxisQueryIterableDataset,
    ExperimentAxisQueryIterDataPipe,
)

assert_array_equal = partial(np.testing.assert_array_equal, strict=True)

# These control which classes are tested (for most, but not all tests).
# Centralized to allow easy add/delete of specific test parameters.
PipeClassType = Union[
    ExperimentAxisQueryIterable,
    ExperimentAxisQueryIterDataPipe,
    ExperimentAxisQueryIterableDataset,
]
PipeClasses = (
    ExperimentAxisQueryIterable,
    ExperimentAxisQueryIterDataPipe,
    ExperimentAxisQueryIterableDataset,
)
XValueGen = Callable[[range, range], spmatrix]


def pytorch_x_value_gen(obs_range: range, var_range: range) -> spmatrix:
    occupied_shape = (
        obs_range.stop - obs_range.start,
        var_range.stop - var_range.start,
    )
    checkerboard_of_ones = coo_matrix(np.indices(occupied_shape).sum(axis=0) % 2)
    checkerboard_of_ones.row += obs_range.start
    checkerboard_of_ones.col += var_range.start
    return checkerboard_of_ones


def pytorch_seq_x_value_gen(obs_range: range, var_range: range) -> spmatrix:
    """A sparse matrix where the values of each col are the obs_range values. Useful for checking the
    X values are being returned in the correct order."""
    data = np.vstack([list(obs_range)] * len(var_range)).flatten()
    rows = np.vstack([list(obs_range)] * len(var_range)).flatten()
    cols = np.column_stack([list(var_range)] * len(obs_range)).flatten()
    return coo_matrix((data, (rows, cols)))
