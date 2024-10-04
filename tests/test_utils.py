# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

import pytest

from tiledbsoma_ml._utils import batched, splits


def test_batched() -> None:

    assert list(batched(range(6), 1)) == list((i,) for i in range(6))
    assert list(batched(range(6), 2)) == [(0, 1), (2, 3), (4, 5)]
    assert list(batched(range(6), 3)) == [(0, 1, 2), (3, 4, 5)]
    assert list(batched(range(6), 4)) == [(0, 1, 2, 3), (4, 5)]
    assert list(batched(range(6), 5)) == [(0, 1, 2, 3, 4), (5,)]
    assert list(batched(range(6), 6)) == [(0, 1, 2, 3, 4, 5)]
    assert list(batched(range(6), 7)) == [(0, 1, 2, 3, 4, 5)]

    # bogus batch value
    with pytest.raises(ValueError):
        list(batched([0, 1], 0))
    with pytest.raises(ValueError):
        list(batched([2, 3], -1))


def test_splits() -> None:
    assert splits(10, 1).tolist() == [0, 10]
    assert splits(10, 2).tolist() == [0, 5, 10]
    assert splits(10, 3).tolist() == [0, 4, 7, 10]
    assert splits(10, 4).tolist() == [0, 3, 6, 8, 10]
    assert splits(10, 10).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert splits(10, 11).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]

    # bad number of sections
    with pytest.raises(ValueError):
        splits(10, 0)
    with pytest.raises(ValueError):
        splits(10, -1)
