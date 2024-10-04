# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

from functools import partial
from typing import Any, Tuple
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from tiledbsoma import Experiment

from tests.utils import pytorch_x_value_gen
from tiledbsoma_ml.dataloader import experiment_dataloader
from tiledbsoma_ml.pytorch import (
    ExperimentAxisQueryIterableDataset,
    ExperimentAxisQueryIterDataPipe,
)


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen", [(6, 3, pytorch_x_value_gen)]
)
@pytest.mark.parametrize(
    "PipeClass",
    (ExperimentAxisQueryIterDataPipe, ExperimentAxisQueryIterableDataset),
)
def test_multiprocessing__returns_full_result(
    PipeClass: ExperimentAxisQueryIterDataPipe | ExperimentAxisQueryIterableDataset,
    soma_experiment: Experiment,
) -> None:
    """Tests the ExperimentAxisQueryIterDataPipe provides all data, as collected from multiple processes that are managed by a
    PyTorch DataLoader with multiple workers configured."""
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        dp = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["soma_joinid", "label"],
            io_batch_size=3,  # two chunks, one per worker
        )
        # Note we're testing the ExperimentAxisQueryIterDataPipe via a DataLoader, since this is what sets up the multiprocessing
        dl = experiment_dataloader(dp, num_workers=2)

        full_result = list(iter(dl))

        soma_joinids = np.concatenate(
            [t[1]["soma_joinid"].to_numpy() for t in full_result]
        )
        assert sorted(soma_joinids) == list(range(6))


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [(3, 3, pytorch_x_value_gen, use_eager_fetch) for use_eager_fetch in (True, False)],
)
@pytest.mark.parametrize(
    "PipeClass",
    (ExperimentAxisQueryIterDataPipe, ExperimentAxisQueryIterableDataset),
)
def test_experiment_dataloader__non_batched(
    PipeClass: ExperimentAxisQueryIterDataPipe | ExperimentAxisQueryIterableDataset,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        dp = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            use_eager_fetch=use_eager_fetch,
        )
        dl = experiment_dataloader(dp)
        data = [row for row in dl]
        assert all(d[0].shape == (3,) for d in data)
        assert all(d[1].shape == (1, 1) for d in data)

        row = data[0]
        assert row[0].tolist() == [0, 1, 0]
        assert row[1]["label"].tolist() == ["0"]


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [(6, 3, pytorch_x_value_gen, use_eager_fetch) for use_eager_fetch in (True, False)],
)
@pytest.mark.parametrize(
    "PipeClass",
    (ExperimentAxisQueryIterDataPipe, ExperimentAxisQueryIterableDataset),
)
def test_experiment_dataloader__batched(
    PipeClass: ExperimentAxisQueryIterDataPipe | ExperimentAxisQueryIterableDataset,
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
        data = [row for row in dl]

        batch = data[0]
        assert batch[0].tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        assert batch[1].to_numpy().tolist() == [[0], [1], [2]]


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [
        (10, 3, pytorch_x_value_gen, use_eager_fetch)
        for use_eager_fetch in (True, False)
    ],
)
@pytest.mark.parametrize(
    "PipeClass",
    (ExperimentAxisQueryIterDataPipe, ExperimentAxisQueryIterableDataset),
)
def test_experiment_dataloader__batched_length(
    PipeClass: ExperimentAxisQueryIterDataPipe | ExperimentAxisQueryIterableDataset,
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
@pytest.mark.parametrize(
    "PipeClass",
    (ExperimentAxisQueryIterDataPipe, ExperimentAxisQueryIterableDataset),
)
def test_experiment_dataloader__collate_fn(
    PipeClass: ExperimentAxisQueryIterDataPipe | ExperimentAxisQueryIterableDataset,
    soma_experiment: Experiment,
    batch_size: int,
) -> None:
    def collate_fn(
        batch_size: int, data: Tuple[npt.NDArray[np.number[Any]], pd.DataFrame]
    ) -> Tuple[npt.NDArray[np.number[Any]], pd.DataFrame]:
        assert isinstance(data, tuple)
        assert len(data) == 2
        assert isinstance(data[0], np.ndarray) and isinstance(data[1], pd.DataFrame)
        if batch_size > 1:
            assert data[0].shape[0] == data[1].shape[0]
            assert data[0].shape[0] <= batch_size
        else:
            assert data[0].ndim == 1
        assert data[1].shape[1] <= batch_size
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

        all_rows = list(iter(dl))
        assert len(all_rows) == 7


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
