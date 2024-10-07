# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Generator, Optional, Sequence, Tuple, Type, Union
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import tiledbsoma as soma
from pandas._testing import assert_frame_equal
from tiledbsoma import Experiment
from utz import parametrize

from tests.conftest import add_dataframe, add_sparse_array
from tests.utils import (
    XValueGen,
    assert_array_equal,
    pytorch_x_value_gen,
)
from tiledbsoma_ml.dataloader import experiment_dataloader
from tiledbsoma_ml.pytorch import (
    ExperimentAxisQueryIterableDataset,
    ExperimentAxisQueryIterDataPipe,
    NDArrayNumber,
    XObsDatum,
)

PipeClassType = Union[
    Type[ExperimentAxisQueryIterDataPipe],
    Type[ExperimentAxisQueryIterableDataset],
]
PipeClasses = (
    ExperimentAxisQueryIterDataPipe,
    ExperimentAxisQueryIterableDataset,
)


@dataclass
class Case:
    obs_range: Union[int, range]
    var_range: Union[int, range]
    PipeClass: PipeClassType = ExperimentAxisQueryIterDataPipe
    X_value_gen: XValueGen = pytorch_x_value_gen
    obsp_layer_names: Optional[Sequence[str]] = None
    varp_layer_names: Optional[Sequence[str]] = None
    measurement_name: str = "RNA"
    X_name: str = "raw"
    obs_column_names: Sequence[str] = ("soma_joinid",)
    batch_size: int = 1
    io_batch_size: int = 2**16
    return_sparse_X: bool = False
    use_eager_fetch: bool = True

    @property
    @contextmanager
    def experiment(self) -> Generator[Experiment, None, None]:
        with TemporaryDirectory() as tmpdir:
            with Experiment.create(tmpdir) as exp:
                if isinstance(self.obs_range, int):
                    obs_range = range(self.obs_range)
                if isinstance(self.var_range, int):
                    var_range = range(self.var_range)

                add_dataframe(exp, "obs", obs_range)
                ms = exp.add_new_collection("ms")
                rna = ms.add_new_collection("RNA", soma.Measurement)
                add_dataframe(rna, "var", var_range)
                rna_x = rna.add_new_collection("X", soma.Collection)
                add_sparse_array(rna_x, "raw", obs_range, var_range, self.X_value_gen)

                if self.obsp_layer_names:
                    obsp = rna.add_new_collection("obsp")
                    for obsp_layer_name in self.obsp_layer_names:
                        add_sparse_array(
                            obsp,
                            obsp_layer_name,
                            obs_range,
                            var_range,
                            self.X_value_gen,
                        )

                if self.varp_layer_names:
                    varp = rna.add_new_collection("varp")
                    for varp_layer_name in self.varp_layer_names:
                        add_sparse_array(
                            varp,
                            varp_layer_name,
                            obs_range,
                            var_range,
                            self.X_value_gen,
                        )
            # Must let the ``Experiment.create`` close, before re-opening / yielding
            yield Experiment.open(tmpdir)

    @property
    @contextmanager
    def datapipe(self) -> Generator[PipeClassType, None, None]:
        with self.experiment as exp:
            with exp.axis_query(measurement_name=self.measurement_name) as query:
                dp = self.PipeClass(
                    query,
                    X_name=self.X_name,
                    obs_column_names=self.obs_column_names,
                    batch_size=self.batch_size,
                    io_batch_size=self.io_batch_size,
                    use_eager_fetch=self.use_eager_fetch,
                    return_sparse_X=self.return_sparse_X,
                )
                yield dp

    @contextmanager
    def dataloader(self, **dataloader_kwargs):
        with self.datapipe as dp:
            dl = experiment_dataloader(dp, **dataloader_kwargs)
            yield dl

    @property
    def batches(self) -> list[XObsDatum]:
        with self.dataloader() as dl:
            return list(dl)

    # Nicer string reprs for test IDs
    _id_fmts = {
        "PipeClass": lambda cls: cls.__name__.replace("ExperimentAxisQuery", ""),
        "use_eager_fetch": lambda b: "eager" if b else "lazy",
    }


@parametrize(
    Case(
        6,
        3,
        obs_column_names=["soma_joinid", "label"],
        io_batch_size=3,  # two chunks, one per worker
    ),
    PipeClass=PipeClasses,
)
def test_multiprocessing__returns_full_result(case: Case):
    """Test that ``ExperimentAxisQueryIter*Data{set,Pipe}`` provides all data, as collected from
    multiple processes managed by a PyTorch DataLoader with multiple workers."""
    with case.dataloader(num_workers=2) as dl:
        batches = list(dl)
    soma_joinids = np.concatenate([t[1]["soma_joinid"].to_numpy() for t in batches])
    assert sorted(soma_joinids) == list(range(6))


@parametrize(
    Case(3, 3, obs_column_names=["label"]),
    PipeClass=PipeClasses,
    use_eager_fetch=[True, False],
)
def test_experiment_dataloader__non_batched(batches):
    for X_batch, obs_df in batches:
        assert X_batch.shape == (3,)
        assert obs_df.shape == (1, 1)

    X_batch, obs_batch = batches[0]
    assert_array_equal(X_batch, np.array([0, 1, 0], dtype=np.float32))
    assert_frame_equal(obs_batch, pd.DataFrame({"label": ["0"]}))


@parametrize(
    Case(6, 3, batch_size=3),
    PipeClass=PipeClasses,
    use_eager_fetch=[True, False],
)
def test_experiment_dataloader__batched(batches):
    X_batch, obs_batch = batches[0]
    assert_array_equal(
        X_batch, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    )
    assert_frame_equal(obs_batch, pd.DataFrame({"soma_joinid": [0, 1, 2]}))


@parametrize(
    Case(10, 3),
    PipeClass=PipeClasses,
    use_eager_fetch=[True, False],
)
def test_experiment_dataloader__batched_length(dataloader):
    with dataloader() as dl:
        assert len(dl) == len(list(dl))


@parametrize(
    Case(10, 3),
    batch_size=[1, 3, 10],
    PipeClass=PipeClasses,
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
    Case(10, 1, obs_column_names=["label"]),
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
