# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

from functools import partial
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from utz import parametrize

from tests.case import Case
from tests.utils import (
    assert_array_equal,
)
from tiledbsoma_ml.dataloader import experiment_dataloader
from tiledbsoma_ml.pytorch import (
    ExperimentAxisQueryIterableDataset,
    ExperimentAxisQueryIterDataPipe,
    NDArrayNumber,
    XObsDatum,
)

# Only test the DataPipe and Dataset classes in this file
# (they each wrap ``ExperimentAxisQueryIterable``)
PipeClasses = (
    ExperimentAxisQueryIterDataPipe,
    ExperimentAxisQueryIterableDataset,
)
sweep_pipeclasses = partial(
    parametrize,
    PipeClass=PipeClasses,
)
sweep_eager_pipeclasses = partial(
    parametrize,
    use_eager_fetch=[True, False],
    PipeClass=PipeClasses,
)


@sweep_pipeclasses(
    Case(
        obs_range=6,
        obs_column_names=["soma_joinid", "label"],
        io_batch_size=3,  # two chunks, one per worker
        num_workers=2,
    )
)
def test_multiprocessing__returns_full_result(batches: list[XObsDatum]):
    """Test that ``ExperimentAxisQueryIter*Data{set,Pipe}`` provides all data, as collected from
    multiple processes managed by a PyTorch DataLoader with multiple workers."""
    soma_joinids = np.concatenate([t[1]["soma_joinid"].to_numpy() for t in batches])
    assert sorted(soma_joinids) == list(range(6))


@sweep_eager_pipeclasses(Case(obs_range=3, obs_column_names=["label"]))
def test_experiment_dataloader__non_batched(batches):
    for X_batch, obs_df in batches:
        assert X_batch.shape == (3,)
        assert obs_df.shape == (1, 1)

    X_batch, obs_batch = batches[0]
    assert_array_equal(X_batch, np.array([0, 1, 0], dtype=np.float32))
    assert_frame_equal(obs_batch, pd.DataFrame({"label": ["0"]}))


@sweep_eager_pipeclasses(Case(obs_range=6, batch_size=3))
def test_experiment_dataloader__batched(batches):
    X_batch, obs_batch = batches[0]
    assert_array_equal(
        X_batch, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    )
    assert_frame_equal(obs_batch, pd.DataFrame({"soma_joinid": [0, 1, 2]}))


@sweep_eager_pipeclasses(Case(obs_range=10))
def test_experiment_dataloader__batched_length(dataloader):
    with dataloader() as dl:
        assert len(dl) == len(list(dl))


@sweep_pipeclasses(
    Case(obs_range=10),
    batch_size=[1, 3, 10],
)
def test_experiment_dataloader__collate_fn(dataloader, batch_size: int):
    def collate_fn(
        data: Tuple[NDArrayNumber, pd.DataFrame]
    ) -> Tuple[NDArrayNumber, pd.DataFrame]:
        assert isinstance(data, tuple)
        assert len(data) == 2
        X_batch, obs_batch = data
        assert isinstance(X_batch, np.ndarray) and isinstance(obs_batch, pd.DataFrame)
        if batch_size > 1:
            assert X_batch.shape[0] == obs_batch.shape[0]
            assert X_batch.shape[0] <= batch_size
        else:
            assert X_batch.ndim == 1
        assert obs_batch.shape[1] <= batch_size
        return data

    with dataloader(collate_fn=collate_fn) as dl:
        batches = list(dl)

    expected_nbatches = {1: 10, 3: 4, 10: 1}[batch_size]
    assert len(batches) == expected_nbatches


@parametrize(
    Case(obs_range=10, var_range=1, obs_column_names=["label"]),
)
def test__pytorch_splitting(datapipe):
    with datapipe as dp:
        # ``random_split`` not available for ``IterableDataset``, yet...
        dp_train, dp_test = dp.random_split(
            weights={"train": 0.7, "test": 0.3}, seed=1234
        )
        dl = experiment_dataloader(dp_train)

        train_batches = list(dl)
        assert len(train_batches) == 7

        dl = experiment_dataloader(dp_test)
        test_batches = list(dl)
        assert len(test_batches) == 3


def test_experiment_dataloader__unsupported_params__fails() -> None:
    with patch(
        "tiledbsoma_ml.pytorch.ExperimentAxisQueryIterDataPipe"
    ) as dummy_exp_data_pipe:
        with pytest.raises(ValueError):
            experiment_dataloader(dummy_exp_data_pipe, shuffle=True)
        with pytest.raises(ValueError):
            experiment_dataloader(dummy_exp_data_pipe, batch_size=3)
        with pytest.raises(ValueError):
            experiment_dataloader(dummy_exp_data_pipe, batch_sampler=[])
        with pytest.raises(ValueError):
            experiment_dataloader(dummy_exp_data_pipe, sampler=[])
