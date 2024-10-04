# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

import itertools
import sys
from itertools import islice
from typing import (
    Iterable,
    Iterator,
    Tuple,
    TypeVar,
)

import numpy as np
import numpy.typing as npt

_T_co = TypeVar("_T_co", covariant=True)


def splits(total_length: int, sections: int) -> npt.NDArray[np.intp]:
    """For ``total_length`` points, compute start/stop offsets that split the length into roughly equal sizes.

    A total_length of L, split into N sections, will return L%N sections of size L//N+1,
    and the remainder as size L//N. This results in the same split as numpy.array_split,
    for an array of length L and sections N.

    Private.

    Examples
    --------
    >>> splits(10, 3)
    array([0,  4,  7, 10])
    >>> splits(4, 2)
    array([0, 2, 4])
    """
    if sections <= 0:
        raise ValueError("number of sections must greater than 0.") from None
    each_section, extras = divmod(total_length, sections)
    per_section_sizes = (
        [0] + extras * [each_section + 1] + (sections - extras) * [each_section]
    )
    splits = np.array(per_section_sizes, dtype=np.intp).cumsum()
    return splits


if sys.version_info >= (3, 12):
    batched = itertools.batched
else:

    def batched(iterable: Iterable[_T_co], n: int) -> Iterator[Tuple[_T_co, ...]]:
        """Same as the Python 3.12+ ``itertools.batched`` -- polyfill for old Python versions."""
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch
