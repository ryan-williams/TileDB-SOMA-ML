# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

from functools import partial
from typing import Tuple, Type, Union
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tiledbsoma import Experiment

from tests.utils import (
    assert_array_equal,
    pytorch_x_value_gen,
    sweep_eager_fetch,
)
from tiledbsoma_ml.dataloader import experiment_dataloader
from tiledbsoma_ml.pytorch import (
    ExperimentAxisQueryIterableDataset,
    ExperimentAxisQueryIterDataPipe,
    NDArrayNumber,
)

PipeClassType = Union[
    Type[ExperimentAxisQueryIterDataPipe],
    Type[ExperimentAxisQueryIterableDataset],
]
PipeClasses = (
    ExperimentAxisQueryIterDataPipe,
    ExperimentAxisQueryIterableDataset,
)


sweep_pipeclasses = pytest.mark.parametrize("PipeClass", PipeClasses)


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen", [(6, 3, pytorch_x_value_gen)]
)
@sweep_pipeclasses
def test_multiprocessing__returns_full_result(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
) -> None:
    """Test that ExperimentAxisQueryIterDataPipe provides all data, as collected from multiple processes managed by a
    PyTorch DataLoader with multiple workers."""
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        dp = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["soma_joinid", "label"],
            io_batch_size=3,  # two chunks, one per worker
        )
        # Test ExperimentAxisQueryIterDataPipe via a DataLoader, since that's what sets up the multiprocessing.
        dl = experiment_dataloader(dp, num_workers=2)

        batches = list(iter(dl))

        soma_joinids = np.concatenate([t[1]["soma_joinid"].to_numpy() for t in batches])
        assert sorted(soma_joinids) == list(range(6))


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,obs_column_names",
    [(3, 3, pytorch_x_value_gen, ["label"])],
)
@sweep_eager_fetch
@sweep_pipeclasses
def test_experiment_dataloader__non_batched(
    PipeClass, use_eager_fetch, soma_experiment, obs_column_names
) -> None:
    with soma_experiment.axis_query("RNA") as query:
        datapipe = PipeClass(
            query,
            obs_column_names=obs_column_names,
            use_eager_fetch=use_eager_fetch,
        )
        dl = experiment_dataloader(datapipe)
        batches = list(dl)
    for X_batch, obs_df in batches:
        assert X_batch.shape == (3,)
        assert obs_df.shape == (1, 1)

    X_batch, obs_batch = batches[0]
    assert_array_equal(X_batch, np.array([0, 1, 0], dtype=np.float32))
    assert_frame_equal(obs_batch, pd.DataFrame({"label": ["0"]}))


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(6, 3, pytorch_x_value_gen)],
)
@sweep_eager_fetch
@sweep_pipeclasses
def test_experiment_dataloader__batched(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        dp = PipeClass(
            query,
            X_name="raw",
            batch_size=3,
            use_eager_fetch=use_eager_fetch,
        )
        dl = experiment_dataloader(dp)
        batches = list(dl)

        X_batch, obs_batch = batches[0]
        assert_array_equal(
            X_batch, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        )
        assert_frame_equal(obs_batch, pd.DataFrame({"soma_joinid": [0, 1, 2]}))


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(10, 3, pytorch_x_value_gen)],
)
@sweep_eager_fetch
@sweep_pipeclasses
def test_experiment_dataloader__batched_length(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        dp = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            use_eager_fetch=use_eager_fetch,
        )
        dl = experiment_dataloader(dp)
        assert len(dl) == len(list(dl))


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,batch_size",
    [(10, 3, pytorch_x_value_gen, batch_size) for batch_size in (1, 3, 10)],
)
@sweep_pipeclasses
def test_experiment_dataloader__collate_fn(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    batch_size: int,
) -> None:
    def collate_fn(
        batch_size: int, data: Tuple[NDArrayNumber, pd.DataFrame]
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

    with soma_experiment.axis_query(measurement_name="RNA") as query:
        dp = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            batch_size=batch_size,
        )
        dl = experiment_dataloader(dp, collate_fn=partial(collate_fn, batch_size))
        assert len(list(dl)) > 0


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen", [(10, 1, pytorch_x_value_gen)]
)
def test__pytorch_splitting(
    soma_experiment: Experiment,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        dp = ExperimentAxisQueryIterDataPipe(
            query,
            X_name="raw",
            obs_column_names=["label"],
        )
        # function not available for IterableDataset, yet....
        dp_train, dp_test = dp.random_split(
            weights={"train": 0.7, "test": 0.3}, seed=1234
        )
        dl = experiment_dataloader(dp_train)

        train_batches = list(iter(dl))
        assert len(train_batches) == 7

        dl = experiment_dataloader(dp_test)
        test_batches = list(iter(dl))
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
