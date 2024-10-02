# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Union
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pytest
import tiledbsoma as soma
from pandas._testing import assert_frame_equal
from scipy import sparse
from scipy.sparse import coo_matrix, spmatrix
from tiledbsoma import Experiment, _factory
from tiledbsoma._collection import CollectionBase
from torch.utils.data._utils.worker import WorkerInfo

from tiledbsoma_ml.pytorch import (
    ExperimentAxisQueryIterable,
    ExperimentAxisQueryIterableDataset,
    ExperimentAxisQueryIterDataPipe,
    experiment_dataloader,
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


@pytest.fixture
def X_layer_names() -> list[str]:
    return ["raw"]


@pytest.fixture
def obsp_layer_names() -> Optional[list[str]]:
    return None


@pytest.fixture
def varp_layer_names() -> Optional[list[str]]:
    return None


def add_dataframe(coll: CollectionBase, key: str, value_range: range) -> None:
    df = coll.add_new_dataframe(
        key,
        schema=pa.schema(
            [
                ("soma_joinid", pa.int64()),
                ("label", pa.large_string()),
                ("label2", pa.large_string()),
            ]
        ),
        index_column_names=["soma_joinid"],
    )
    df.write(
        pa.Table.from_pydict(
            {
                "soma_joinid": list(value_range),
                "label": [str(i) for i in value_range],
                "label2": ["c" for i in value_range],
            }
        )
    )


def add_sparse_array(
    coll: CollectionBase,
    key: str,
    obs_range: range,
    var_range: range,
    value_gen: XValueGen,
) -> None:
    a = coll.add_new_sparse_ndarray(
        key, type=pa.float32(), shape=(obs_range.stop, var_range.stop)
    )
    tensor = pa.SparseCOOTensor.from_scipy(value_gen(obs_range, var_range))
    a.write(tensor)


@pytest.fixture(scope="function")
def soma_experiment(
    tmp_path: Path,
    obs_range: Union[int, range],
    var_range: Union[int, range],
    X_value_gen: XValueGen,
    obsp_layer_names: Sequence[str],
    varp_layer_names: Sequence[str],
) -> soma.Experiment:
    with soma.Experiment.create((tmp_path / "exp").as_posix()) as exp:
        if isinstance(obs_range, int):
            obs_range = range(obs_range)
        if isinstance(var_range, int):
            var_range = range(var_range)

        add_dataframe(exp, "obs", obs_range)
        ms = exp.add_new_collection("ms")
        rna = ms.add_new_collection("RNA", soma.Measurement)
        add_dataframe(rna, "var", var_range)
        rna_x = rna.add_new_collection("X", soma.Collection)
        add_sparse_array(rna_x, "raw", obs_range, var_range, X_value_gen)

        if obsp_layer_names:
            obsp = rna.add_new_collection("obsp")
            for obsp_layer_name in obsp_layer_names:
                add_sparse_array(
                    obsp, obsp_layer_name, obs_range, var_range, X_value_gen
                )

        if varp_layer_names:
            varp = rna.add_new_collection("varp")
            for varp_layer_name in varp_layer_names:
                add_sparse_array(
                    varp, varp_layer_name, obs_range, var_range, X_value_gen
                )
    return _factory.open((tmp_path / "exp").as_posix())


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(6, 3, pytorch_x_value_gen)],
)
@pytest.mark.parametrize("use_eager_fetch", [True, False])
@pytest.mark.parametrize("return_sparse_X", [True, False])
@pytest.mark.parametrize("PipeClass", PipeClasses)
def test_non_batched(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
    return_sparse_X: bool,
) -> None:
    """Check batches of size 1 (the default)"""
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        exp_data_pipe = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            shuffle=False,
            use_eager_fetch=use_eager_fetch,
            return_sparse_X=return_sparse_X,
        )
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


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(6, 3, pytorch_x_value_gen)],
)
@pytest.mark.parametrize("use_eager_fetch", [True, False])
@pytest.mark.parametrize("return_sparse_X", [True, False])
@pytest.mark.parametrize("PipeClass", PipeClasses)
def test_uneven_soma_and_result_batches(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
    return_sparse_X: bool,
) -> None:
    """Check that batches are correctly created when they require fetching multiple chunks."""
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        exp_data_pipe = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            shuffle=False,
            batch_size=3,
            io_batch_size=2,
            use_eager_fetch=use_eager_fetch,
            return_sparse_X=return_sparse_X,
        )
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


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(6, 3, pytorch_x_value_gen)],
)
@pytest.mark.parametrize("use_eager_fetch", [True, False])
@pytest.mark.parametrize("return_sparse_X", [True, False])
@pytest.mark.parametrize("PipeClass", PipeClasses)
def test_batching__all_batches_full_size(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
    return_sparse_X: bool,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        exp_data_pipe = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            shuffle=False,
            use_eager_fetch=use_eager_fetch,
            return_sparse_X=return_sparse_X,
        )
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


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(range(100_000_000, 100_000_003), 3, pytorch_x_value_gen)],
)
@pytest.mark.parametrize("use_eager_fetch", [True, False])
@pytest.mark.parametrize("PipeClass", PipeClasses)
def test_soma_joinids(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        exp_data_pipe = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["soma_joinid", "label"],
            batch_size=3,
            shuffle=False,
            use_eager_fetch=use_eager_fetch,
        )
        assert exp_data_pipe.shape == (1, 3)

        soma_joinids = np.concatenate(
            [batch[1]["soma_joinid"].to_numpy() for batch in exp_data_pipe]
        )
        assert_array_equal(soma_joinids, np.arange(100_000_000, 100_000_003))


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(5, 3, pytorch_x_value_gen)],
)
@pytest.mark.parametrize("use_eager_fetch", [True, False])
@pytest.mark.parametrize("return_sparse_X", [True, False])
@pytest.mark.parametrize("PipeClass", PipeClasses)
def test_batching__partial_final_batch_size(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
    return_sparse_X: bool,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        exp_data_pipe = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            shuffle=False,
            use_eager_fetch=use_eager_fetch,
            return_sparse_X=return_sparse_X,
        )
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


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(3, 3, pytorch_x_value_gen)],
)
@pytest.mark.parametrize("use_eager_fetch", [True, False])
@pytest.mark.parametrize("PipeClass", PipeClasses)
def test_batching__exactly_one_batch(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        exp_data_pipe = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            shuffle=False,
            use_eager_fetch=use_eager_fetch,
        )
        assert exp_data_pipe.shape == (1, 3)
        batch_iter = iter(exp_data_pipe)
        X_batch, obs_batch = next(batch_iter)
        assert X_batch.tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        assert_frame_equal(obs_batch, pd.DataFrame({"label": ["0", "1", "2"]}))

        with pytest.raises(StopIteration):
            next(batch_iter)


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen",
    [(6, 3, pytorch_x_value_gen)],
)
@pytest.mark.parametrize("use_eager_fetch", [True, False])
@pytest.mark.parametrize("PipeClass", PipeClasses)
def test_batching__empty_query_result(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    use_eager_fetch: bool,
) -> None:
    with soma_experiment.axis_query(
        measurement_name="RNA", obs_query=soma.AxisQuery(coords=([],))
    ) as query:
        exp_data_pipe = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            use_eager_fetch=use_eager_fetch,
        )
        assert exp_data_pipe.shape == (0, 3)
        batch_iter = iter(exp_data_pipe)

        with pytest.raises(StopIteration):
            next(batch_iter)


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [
        (10, 1, pytorch_x_value_gen, use_eager_fetch)
        for use_eager_fetch in (True, False)
    ],
)
@pytest.mark.parametrize("PipeClass", PipeClasses)
def test_batching__partial_soma_batches_are_concatenated(
    PipeClass: PipeClassType, soma_experiment: Experiment, use_eager_fetch: bool
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        exp_data_pipe = PipeClass(
            query,
            X_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            # set SOMA batch read size such that PyTorch batches will span the tail and head of two SOMA batches
            io_batch_size=4,
            use_eager_fetch=use_eager_fetch,
        )

        batches = list(exp_data_pipe)

        assert [len(batch[0]) for batch in batches] == [3, 3, 3, 1]


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
    "obs_range,var_range,X_value_gen",
    [(6, 3, pytorch_x_value_gen), (7, 3, pytorch_x_value_gen)],
)
@pytest.mark.parametrize(
    "world_size,rank",
    [(3, 0), (3, 1), (3, 2), (2, 0), (2, 1)],
)
@pytest.mark.parametrize("PipeClass", PipeClasses)
def test_distributed__returns_data_partition_for_rank(
    PipeClass: PipeClassType,
    soma_experiment: Experiment,
    obs_range: int,
    world_size: int,
    rank: int,
) -> None:
    """Tests pytorch._partition_obs_joinids() behavior in a simulated PyTorch distributed processing mode,
    using mocks to avoid having to do real PyTorch distributed setup."""

    with (
        patch("torch.distributed.is_initialized") as mock_dist_is_initialized,
        patch("torch.distributed.get_rank") as mock_dist_get_rank,
        patch("torch.distributed.get_world_size") as mock_dist_get_world_size,
    ):
        mock_dist_is_initialized.return_value = True
        mock_dist_get_rank.return_value = rank
        mock_dist_get_world_size.return_value = world_size

        with soma_experiment.axis_query(measurement_name="RNA") as query:
            dp = PipeClass(
                query,
                X_name="raw",
                obs_column_names=["soma_joinid"],
                io_batch_size=2,
                shuffle=False,
            )
            batches = list(iter(dp))
            soma_joinids = np.concatenate(
                [batch[1]["soma_joinid"].to_numpy() for batch in batches]
            )

            expected_joinids = np.array_split(np.arange(obs_range), world_size)[rank][
                0 : obs_range // world_size
            ].tolist()
            assert sorted(soma_joinids) == expected_joinids


# fmt: off
@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,world_size,num_workers,splits",
    [
        (12, 3, pytorch_x_value_gen, 3, 2, [[0, 2, 4], [4,  6,  8], [ 8, 10, 12]]),
        (13, 3, pytorch_x_value_gen, 3, 2, [[0, 2, 4], [5,  7,  9], [ 9, 11, 13]]),
        (15, 3, pytorch_x_value_gen, 3, 2, [[0, 4, 5], [5,  9, 10], [10, 14, 15]]),
        (16, 3, pytorch_x_value_gen, 3, 2, [[0, 4, 5], [6, 10, 11], [11, 15, 16]]),
        (18, 3, pytorch_x_value_gen, 3, 2, [[0, 4, 6], [6, 10, 12], [12, 16, 18]]),
        (19, 3, pytorch_x_value_gen, 3, 2, [[0, 4, 6], [7, 11, 13], [13, 17, 19]]),
        (20, 3, pytorch_x_value_gen, 3, 2, [[0, 4, 6], [7, 11, 13], [14, 18, 20]]),
        (21, 3, pytorch_x_value_gen, 3, 2, [[0, 4, 7], [7, 11, 14], [14, 18, 21]]),
        (25, 3, pytorch_x_value_gen, 3, 2, [[0, 4, 8], [9, 13, 17], [17, 21, 25]]),
        (27, 3, pytorch_x_value_gen, 3, 2, [[0, 6, 9], [9, 15, 18], [18, 24, 27]]),
    ],
)
# fmt: on
def test_distributed_and_multiprocessing__returns_data_partition_for_rank(
    soma_experiment: Experiment,
    world_size: int,
    num_workers: int,
    splits: list[list[int]],
) -> None:
    """Tests pytorch._partition_obs_joinids() behavior in a simulated PyTorch distributed processing mode and
    DataLoader multiprocessing mode, using mocks to avoid having to do distributed pytorch
    setup or real DataLoader multiprocessing."""

    for rank in range(world_size):
        proc_splits = splits[rank]
        for worker_id in range(num_workers):
            expected_joinids = list(
                range(proc_splits[worker_id], proc_splits[worker_id + 1])
            )
            with (
                patch("torch.utils.data.get_worker_info") as mock_get_worker_info,
                patch("torch.distributed.is_initialized") as mock_dist_is_initialized,
                patch("torch.distributed.get_rank") as mock_dist_get_rank,
                patch("torch.distributed.get_world_size") as mock_dist_get_world_size,
            ):
                mock_get_worker_info.return_value = WorkerInfo(
                    id=worker_id, num_workers=num_workers, seed=1234
                )
                mock_dist_is_initialized.return_value = True
                mock_dist_get_rank.return_value = rank
                mock_dist_get_world_size.return_value = world_size

                with soma_experiment.axis_query(measurement_name="RNA") as query:
                    dp = ExperimentAxisQueryIterable(
                        query,
                        X_name="raw",
                        obs_column_names=["soma_joinid"],
                        io_batch_size=2,
                        shuffle=False,
                    )

                    batches = list(iter(dp))

                    soma_joinids = np.concatenate(
                        [batch[1]["soma_joinid"].to_numpy() for batch in batches]
                    ).tolist()

                    assert soma_joinids == expected_joinids


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
            shuffle=False,
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
            shuffle=False,
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
            shuffle=False,
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
            shuffle=False,
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


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen", [(16, 1, pytorch_seq_x_value_gen)]
)
@pytest.mark.parametrize("PipeClass", PipeClasses)
def test__shuffle(PipeClass: PipeClassType, soma_experiment: Experiment) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        dp = PipeClass(
            query,
            X_name="raw",
            shuffle=True,
        )

        all_rows = list(iter(dp))
        if PipeClass is ExperimentAxisQueryIterable:
            assert all(np.squeeze(r[0], axis=0).shape == (1,) for r in all_rows)
        else:
            assert all(r[0].shape == (1,) for r in all_rows)
        soma_joinids = [row[1]["soma_joinid"].iloc[0] for row in all_rows]
        X_values = [row[0][0].item() for row in all_rows]

        # same elements
        assert set(soma_joinids) == set(range(16))
        # not ordered! (...with a `1/16!` probability of being ordered)
        assert soma_joinids != list(range(16))
        # randomizes X in same order as obs
        # note: X values were explicitly set to match obs_joinids to allow for this simple assertion
        assert X_values == soma_joinids


@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen", [(6, 3, pytorch_x_value_gen)]
)
def test_experiment_axis_query_iterable_error_checks(
    soma_experiment: Experiment,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        dp = ExperimentAxisQueryIterable(
            query,
            X_name="raw",
            shuffle=True,
        )
        with pytest.raises(NotImplementedError):
            dp[0]

        with pytest.raises(ValueError):
            dp = ExperimentAxisQueryIterable(
                query,
                obs_column_names=(),
                X_name="raw",
                shuffle=True,
            )


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


def test_batched() -> None:
    from tiledbsoma_ml.pytorch import _batched

    assert list(_batched(range(6), 1)) == list((i,) for i in range(6))
    assert list(_batched(range(6), 2)) == [(0, 1), (2, 3), (4, 5)]
    assert list(_batched(range(6), 3)) == [(0, 1, 2), (3, 4, 5)]
    assert list(_batched(range(6), 4)) == [(0, 1, 2, 3), (4, 5)]
    assert list(_batched(range(6), 5)) == [(0, 1, 2, 3, 4), (5,)]
    assert list(_batched(range(6), 6)) == [(0, 1, 2, 3, 4, 5)]
    assert list(_batched(range(6), 7)) == [(0, 1, 2, 3, 4, 5)]

    # bogus batch value
    with pytest.raises(ValueError):
        list(_batched([0, 1], 0))
    with pytest.raises(ValueError):
        list(_batched([2, 3], -1))


def test_splits() -> None:
    from tiledbsoma_ml.pytorch import _splits

    assert _splits(10, 1).tolist() == [0, 10]
    assert _splits(10, 2).tolist() == [0, 5, 10]
    assert _splits(10, 3).tolist() == [0, 4, 7, 10]
    assert _splits(10, 4).tolist() == [0, 3, 6, 8, 10]
    assert _splits(10, 10).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert _splits(10, 11).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]

    # bad number of sections
    with pytest.raises(ValueError):
        _splits(10, 0)
    with pytest.raises(ValueError):
        _splits(10, -1)


@pytest.mark.parametrize(  # keep these small as we materialize as a dense ndarray
    "shape",
    [(100, 10), (10, 100), (1, 1), (1, 100), (100, 1), (0, 0), (10, 0), (0, 10)],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_csr__construct_from_ijd(shape: Tuple[int, int], dtype: npt.DTypeLike) -> None:
    from tiledbsoma_ml.pytorch import _CSR_IO_Buffer

    sp_coo = sparse.random(shape[0], shape[1], dtype=dtype, format="coo", density=0.05)
    sp_csr = sp_coo.tocsr()

    _ncsr = _CSR_IO_Buffer.from_ijd(
        sp_coo.row, sp_coo.col, sp_coo.data, shape=sp_coo.shape
    )
    assert _ncsr.nnz == sp_coo.nnz == sp_csr.nnz
    assert _ncsr.dtype == sp_coo.dtype == sp_csr.dtype
    assert _ncsr.nbytes == (
        _ncsr.data.nbytes + _ncsr.indices.nbytes + _ncsr.indptr.nbytes
    )

    # _CSR_IO_Buffer makes no guarantees about minor axis ordering (ie, "canonical" form) until
    # sort_indices is called, so use the SciPy sparse csr package to validate by round-tripping.
    assert (
        sparse.csr_matrix((_ncsr.data, _ncsr.indices, _ncsr.indptr), shape=_ncsr.shape)
        != sp_csr
    ).nnz == 0

    # Check dense slicing
    assert np.array_equal(_ncsr.slice_tonumpy(slice(0, shape[0])), sp_coo.toarray())
    assert np.array_equal(_ncsr.slice_tonumpy(slice(0, shape[0])), sp_csr.toarray())
    assert np.array_equal(_ncsr.slice_tonumpy(slice(1, -1)), sp_csr[1:-1].toarray())
    assert np.array_equal(_ncsr.slice_tonumpy(slice(None, -2)), sp_csr[:-2].toarray())
    assert np.array_equal(_ncsr.slice_tonumpy(slice(None)), sp_csr[:].toarray())

    # Check sparse slicing
    assert (_ncsr.slice_toscipy(slice(0, shape[0])) != sp_csr).nnz == 0
    assert (_ncsr.slice_toscipy(slice(1, -1)) != sp_csr[1:-1]).nnz == 0
    assert (_ncsr.slice_toscipy(slice(None, -2)) != sp_csr[:-2]).nnz == 0
    assert (_ncsr.slice_toscipy(slice(None)) != sp_csr[:]).nnz == 0


@pytest.mark.parametrize(
    "shape",
    [(100, 10), (10, 100), (1, 1), (1, 100), (100, 1), (0, 0), (10, 0), (0, 10)],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_csr__construct_from_pjd(shape: Tuple[int, int], dtype: npt.DTypeLike) -> None:
    from tiledbsoma_ml.pytorch import _CSR_IO_Buffer

    sp_csr = sparse.random(shape[0], shape[1], dtype=dtype, format="csr", density=0.05)

    _ncsr = _CSR_IO_Buffer.from_pjd(
        sp_csr.indptr.copy(),
        sp_csr.indices.copy(),
        sp_csr.data.copy(),
        shape=sp_csr.shape,
    )

    # _CSR_IO_Buffer makes no guarantees about minor axis ordering (ie, "canonical" form) until
    # sort_indices is called, so use the SciPy sparse csr package to validate by round-tripping.
    assert (
        sparse.csr_matrix((_ncsr.data, _ncsr.indices, _ncsr.indptr), shape=_ncsr.shape)
        != sp_csr
    ).nnz == 0

    # Check dense slicing
    assert np.array_equal(_ncsr.slice_tonumpy(slice(0, shape[0])), sp_csr.toarray())
    assert np.array_equal(_ncsr.slice_tonumpy(slice(1, -1)), sp_csr[1:-1].toarray())
    assert np.array_equal(_ncsr.slice_tonumpy(slice(None, -2)), sp_csr[:-2].toarray())
    assert np.array_equal(_ncsr.slice_tonumpy(slice(None)), sp_csr[:].toarray())

    # Check sparse slicing
    assert (_ncsr.slice_toscipy(slice(0, shape[0])) != sp_csr).nnz == 0
    assert (_ncsr.slice_toscipy(slice(1, -1)) != sp_csr[1:-1]).nnz == 0
    assert (_ncsr.slice_toscipy(slice(None, -2)) != sp_csr[:-2]).nnz == 0
    assert (_ncsr.slice_toscipy(slice(None)) != sp_csr[:]).nnz == 0


@pytest.mark.parametrize(
    "shape",
    [(100, 10), (10, 100)],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
@pytest.mark.parametrize("n_splits", [2, 3, 4])
def test_csr__merge(
    shape: Tuple[int, int], dtype: npt.DTypeLike, n_splits: int
) -> None:
    from tiledbsoma_ml.pytorch import _CSR_IO_Buffer

    sp_coo = sparse.random(shape[0], shape[1], dtype=dtype, format="coo", density=0.5)
    splits = [
        t
        for t in zip(
            np.array_split(sp_coo.row, n_splits),
            np.array_split(sp_coo.col, n_splits),
            np.array_split(sp_coo.data, n_splits),
            strict=False,
        )
    ]
    _ncsr = _CSR_IO_Buffer.merge(
        [_CSR_IO_Buffer.from_ijd(i, j, d, shape=sp_coo.shape) for i, j, d in splits]
    )

    assert (
        sp_coo.tocsr()
        != sparse.csr_matrix(
            (_ncsr.data, _ncsr.indices, _ncsr.indptr), shape=_ncsr.shape
        )
    ).nnz == 0


@pytest.mark.parametrize(
    "shape",
    [(100, 10), (10, 100), (1, 1), (1, 100), (100, 1), (0, 0), (10, 0), (0, 10)],
)
def test_csr__sort_indices(shape: Tuple[int, int]) -> None:
    from tiledbsoma_ml.pytorch import _CSR_IO_Buffer

    sp_coo = sparse.random(
        shape[0], shape[1], dtype=np.float32, format="coo", density=0.05
    )
    sp_csr = sp_coo.tocsr()

    _ncsr = _CSR_IO_Buffer.from_ijd(
        sp_coo.row, sp_coo.col, sp_coo.data, shape=sp_coo.shape
    ).sort_indices()

    assert np.array_equal(sp_csr.indptr, _ncsr.indptr)
    assert np.array_equal(sp_csr.indices, _ncsr.indices)
    assert np.array_equal(sp_csr.data, _ncsr.data)
