# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

from functools import partial
from typing import ContextManager, Union
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from scipy import sparse
from tiledbsoma import AxisQuery
from torch.utils.data._utils.worker import WorkerInfo
from utz import parametrize

from tests.case import Case, PipeClassType
from tests.utils import (
    assert_array_equal,
)
from tiledbsoma_ml.pytorch import (
    ExperimentAxisQueryIterable,
    ExperimentAxisQueryIterableDataset,
    ExperimentAxisQueryIterDataPipe,
    XObsDatum,
)

# Classes which are exercised in most test cases.
PipeClass = Union[
    ExperimentAxisQueryIterable,
    ExperimentAxisQueryIterDataPipe,
    ExperimentAxisQueryIterableDataset,
]
PipeClasses = (
    ExperimentAxisQueryIterable,
    ExperimentAxisQueryIterDataPipe,
    ExperimentAxisQueryIterableDataset,
)


sweep_all = partial(
    parametrize,
    use_eager_fetch=[True, False],
    return_sparse_X=[True, False],
    PipeClass=PipeClasses,
)
sweep_eager_pipeclasses = partial(
    parametrize,
    use_eager_fetch=[True, False],
    PipeClass=PipeClasses,
)
sweep_pipeclasses = partial(parametrize, PipeClass=PipeClasses)


@sweep_all(Case(obs_range=6, obs_column_names=["label"]))
def test_non_batched(
    datapipe: ContextManager[PipeClass],
    PipeClass: PipeClassType,
    return_sparse_X: bool,
):
    """Check batches of size 1 (the default)"""
    with datapipe as exp_data_pipe:
        assert exp_data_pipe.shape == (6, 3)
        batch_iter = iter(exp_data_pipe)
        for idx, (X_batch, obs_batch) in enumerate(batch_iter):
            expected_X = [0, 1, 0] if idx % 2 == 0 else [1, 0, 1]
            if return_sparse_X:
                assert isinstance(X_batch, sparse.csr_matrix)
                # Sparse slices are always 2D
                assert X_batch.shape == (1, 3)
                assert X_batch.todense().tolist() == [expected_X]
            else:
                assert isinstance(X_batch, np.ndarray)
                if PipeClass is ExperimentAxisQueryIterable:
                    assert X_batch.shape == (1, 3)
                    assert X_batch.tolist() == [expected_X]
                else:
                    # ExperimentAxisQueryIterData{Pipe,set} "squeeze" dense single-row batches
                    assert X_batch.shape == (3,)
                    assert X_batch.tolist() == expected_X

            assert_frame_equal(obs_batch, pd.DataFrame({"label": [str(idx)]}))


@sweep_all(
    Case(
        obs_range=6,
        obs_column_names=["label"],
        batch_size=3,
        io_batch_size=2,
    )
)
def test_uneven_soma_and_result_batches(
    datapipe: ContextManager[PipeClass],
    return_sparse_X: bool,
):
    """Check that batches are correctly created when they require fetching multiple chunks."""
    with datapipe as exp_data_pipe:
        assert exp_data_pipe.shape == (2, 3)
        batch_iter = iter(exp_data_pipe)

        X_batch, obs_batch = next(batch_iter)
        assert X_batch.shape == (3, 3)
        if return_sparse_X:
            assert isinstance(X_batch, sparse.csr_matrix)
            X_batch = X_batch.todense()
        else:
            assert isinstance(X_batch, np.ndarray)
        assert X_batch.tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        assert_frame_equal(obs_batch, pd.DataFrame({"label": ["0", "1", "2"]}))

        X_batch, obs_batch = next(batch_iter)
        assert X_batch.shape == (3, 3)
        if return_sparse_X:
            assert isinstance(X_batch, sparse.csr_matrix)
            X_batch = X_batch.todense()
        else:
            assert isinstance(X_batch, np.ndarray)
        assert X_batch.tolist() == [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        assert_frame_equal(obs_batch, pd.DataFrame({"label": ["3", "4", "5"]}))


@sweep_all(Case(obs_range=6, obs_column_names=["label"], batch_size=3))
def test_batching__all_batches_full_size(
    datapipe: ContextManager[PipeClass],
    return_sparse_X: bool,
):
    with datapipe as exp_data_pipe:
        batch_iter = iter(exp_data_pipe)
        assert exp_data_pipe.shape == (2, 3)

        X_batch, obs_batch = next(batch_iter)
        if return_sparse_X:
            assert isinstance(X_batch, sparse.csr_matrix)
            X_batch = X_batch.todense()
        assert X_batch.tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        assert_frame_equal(obs_batch, pd.DataFrame({"label": ["0", "1", "2"]}))

        X_batch, obs_batch = next(batch_iter)
        if return_sparse_X:
            assert isinstance(X_batch, sparse.csr_matrix)
            X_batch = X_batch.todense()
        assert X_batch.tolist() == [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        assert_frame_equal(obs_batch, pd.DataFrame({"label": ["3", "4", "5"]}))

        with pytest.raises(StopIteration):
            next(batch_iter)


@sweep_eager_pipeclasses(
    Case(
        obs_range=range(100_000_000, 100_000_003),
        obs_column_names=["soma_joinid", "label"],
        batch_size=3,
    )
)
def test_soma_joinids(datapipe: ContextManager[PipeClass]):
    with datapipe as exp_data_pipe:
        assert exp_data_pipe.shape == (1, 3)

        soma_joinids = np.concatenate(
            [batch[1]["soma_joinid"].to_numpy() for batch in exp_data_pipe]
        )
        assert_array_equal(soma_joinids, np.arange(100_000_000, 100_000_003))


@sweep_all(Case(obs_range=5, obs_column_names=["label"], batch_size=3))
def test_batching__partial_final_batch_size(
    datapipe: ContextManager[PipeClass],
    return_sparse_X: bool,
):
    with datapipe as exp_data_pipe:
        assert exp_data_pipe.shape == (2, 3)
        batch_iter = iter(exp_data_pipe)

        next(batch_iter)
        X_batch, obs_batch = next(batch_iter)
        if return_sparse_X:
            assert isinstance(X_batch, sparse.csr_matrix)
            X_batch = X_batch.todense()
        assert X_batch.tolist() == [[1, 0, 1], [0, 1, 0]]
        assert_frame_equal(obs_batch, pd.DataFrame({"label": ["3", "4"]}))

        with pytest.raises(StopIteration):
            next(batch_iter)


@sweep_eager_pipeclasses(Case(obs_range=3, obs_column_names=["label"], batch_size=3))
def test_batching__exactly_one_batch(datapipe: ContextManager[PipeClass]):
    with datapipe as exp_data_pipe:
        assert exp_data_pipe.shape == (1, 3)
        batch_iter = iter(exp_data_pipe)
        X_batch, obs_batch = next(batch_iter)
        assert X_batch.tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        assert_frame_equal(obs_batch, pd.DataFrame({"label": ["0", "1", "2"]}))

        with pytest.raises(StopIteration):
            next(batch_iter)


@sweep_eager_pipeclasses(
    Case(
        obs_range=6,
        batch_size=3,
        obs_column_names=["label"],
        obs_query=AxisQuery(coords=([],)),
    ),
)
def test_batching__empty_query_result(datapipe: ContextManager[PipeClass]):
    with datapipe as exp_data_pipe:
        assert exp_data_pipe.shape == (0, 3)
        batch_iter = iter(exp_data_pipe)

        with pytest.raises(StopIteration):
            next(batch_iter)


@sweep_eager_pipeclasses(
    Case(
        obs_range=10,
        var_range=1,
        obs_column_names=["label"],
        batch_size=3,
        # Set SOMA batch read size such that PyTorch batches will span the tail and head of two SOMA batches
        io_batch_size=4,
    )
)
def test_batching__partial_soma_batches_are_concatenated(batches: list[XObsDatum]):
    assert [len(batch[0]) for batch in batches] == [3, 3, 3, 1]


@sweep_pipeclasses(
    [
        Case(
            obs_range=6,
            obs_column_names=["soma_joinid"],
            io_batch_size=2,
            world_size=world_size,
            rank=rank,
        )
        for world_size, rank in [(3, 0), (3, 1), (3, 2), (2, 0), (2, 1)]
    ],
    obs_range=[6, 7],
)
def test_distributed__returns_data_partition_for_rank(
    datapipe: ContextManager[PipeClass],
    obs_range: int,
    init_world: ContextManager,
    world_size: int,
    rank: int,
):
    """Tests pytorch._partition_obs_joinids() behavior in a simulated PyTorch distributed processing mode,
    using mocks to avoid having to do real PyTorch distributed setup."""
    with init_world, datapipe as dp:
        batches = list(dp)
        soma_joinids = np.concatenate(
            [batch[1]["soma_joinid"].to_numpy() for batch in batches]
        )

        expected_joinids = np.array_split(np.arange(obs_range), world_size)[rank][
            0 : obs_range // world_size
        ].tolist()
        assert sorted(soma_joinids) == expected_joinids


# fmt: off
@parametrize([
    Case(
        obs_range=obs_range,
        PipeClass=ExperimentAxisQueryIterable,
        obs_column_names=["soma_joinid"],
        io_batch_size=2,
        world_size=3,
        num_workers=2,
        expected_splits=expected_splits,
    ) for obs_range, expected_splits in [
        (12, [[0, 2, 4], [4,  6,  8], [ 8, 10, 12]]),
        (13, [[0, 2, 4], [5,  7,  9], [ 9, 11, 13]]),
        (15, [[0, 4, 5], [5,  9, 10], [10, 14, 15]]),
        (16, [[0, 4, 5], [6, 10, 11], [11, 15, 16]]),
        (18, [[0, 4, 6], [6, 10, 12], [12, 16, 18]]),
        (19, [[0, 4, 6], [7, 11, 13], [13, 17, 19]]),
        (20, [[0, 4, 6], [7, 11, 13], [14, 18, 20]]),
        (21, [[0, 4, 7], [7, 11, 14], [14, 18, 21]]),
        (25, [[0, 4, 8], [9, 13, 17], [17, 21, 25]]),
        (27, [[0, 6, 9], [9, 15, 18], [18, 24, 27]]),
    ]],
    rank=[0, 1, 2],
    worker_id=[0, 1],
)
# fmt: on
def test_distributed_and_multiprocessing__returns_data_partition_for_rank(
    init_world: ContextManager,
    datapipe: ContextManager[PipeClass],
    rank: int,
    num_workers: int,
    worker_id: int,
    expected_splits: list[list[int]],
):
    """Tests pytorch._partition_obs_joinids() behavior in a simulated PyTorch distributed processing mode and
    DataLoader multiprocessing mode, using mocks to avoid having to do distributed pytorch
    setup or real DataLoader multiprocessing."""

    proc_splits = expected_splits[rank]
    expected_joinids = list(range(proc_splits[worker_id], proc_splits[worker_id + 1]))
    with (
        init_world,
        patch("torch.utils.data.get_worker_info") as mock_get_worker_info,
    ):
        mock_get_worker_info.return_value = WorkerInfo(
            id=worker_id, num_workers=num_workers, seed=1234
        )

        with datapipe as dp:
            batches = list(dp)

            soma_joinids = np.concatenate(
                [batch[1]["soma_joinid"].to_numpy() for batch in batches]
            ).tolist()

            assert soma_joinids == expected_joinids
