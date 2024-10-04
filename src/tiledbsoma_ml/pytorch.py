# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

import contextlib
import gc
import itertools
import logging
import os
import sys
import time
from contextlib import contextmanager
from itertools import islice
from math import ceil
from typing import (
    Any,
    ContextManager,
    Dict,
    Generator,
    Iterable,
    Iterator,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import attrs
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sparse
import tiledbsoma as soma
import torch
import torchdata
from somacore.query._eager_iter import EagerIterator as _EagerIterator

logger = logging.getLogger("tiledbsoma_ml.pytorch")

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)

NDArrayNumber = npt.NDArray[np.number[Any]]
XDatum = Union[NDArrayNumber, sparse.csr_matrix]
XObsDatum = Tuple[XDatum, pd.DataFrame]
"""Return type of ``ExperimentAxisQueryIterableDataset`` and ``ExperimentAxisQueryIterDataPipe``,
which pairs a slice of ``X`` rows with a corresponding slice of ``obs``. In the default case,
the datum is a tuple of :class:`numpy.ndarray` and :class:`pandas.DataFrame` (for ``X`` and ``obs``
respectively). If the object is created with ``return_sparse_X`` as True, the ``X`` slice is
returned as a :class:`scipy.sparse.csr_matrix`. If the ``batch_size`` is 1, the :class:`numpy.ndarray`
will be returned with rank 1; in all other cases, objects are returned with rank 2."""


@attrs.define(frozen=True, kw_only=True)
class _ExperimentLocator:
    """State required to open the Experiment.

    Serializable across multiple processes.

    Private implementation class.
    """

    uri: str
    tiledb_timestamp_ms: int
    tiledb_config: Dict[str, Union[str, float]]

    @classmethod
    def create(cls, experiment: soma.Experiment) -> "_ExperimentLocator":
        return _ExperimentLocator(
            uri=experiment.uri,
            tiledb_timestamp_ms=experiment.tiledb_timestamp_ms,
            tiledb_config=experiment.context.tiledb_config,
        )

    @contextmanager
    def open_experiment(self) -> Generator[soma.Experiment, None, None]:
        context = soma.SOMATileDBContext(tiledb_config=self.tiledb_config)
        yield soma.Experiment.open(
            self.uri, tiledb_timestamp=self.tiledb_timestamp_ms, context=context
        )


class ExperimentAxisQueryIterable(Iterable[XObsDatum]):
    """An :class:`Iterable` which reads ``X`` and ``obs`` data from a :class:`tiledbsoma.Experiment`, as
    selected by a user-specified :class:`tiledbsoma.ExperimentAxisQuery`. Each step of the iterator
    produces a batch containing equal-sized ``X`` and ``obs`` data, in the form of a :class:`numpy.ndarray` and
    :class:`pandas.DataFrame`, respectively.

    Private base class for subclasses of :class:`torch.utils.data.IterableDataset` and
    :class:`torchdata.datapipes.iter.IterDataPipe`. Refer to :class:`ExperimentAxisQueryIterableDataset`
    and :class:`ExperimentAxisQueryIterDataPipe` for more details on usage.

    Lifecycle:
        experimental
    """

    def __init__(
        self,
        query: soma.ExperimentAxisQuery,
        X_name: str,
        obs_column_names: Sequence[str] = ("soma_joinid",),
        batch_size: int = 1,
        io_batch_size: int = 2**16,
        return_sparse_X: bool = False,
        use_eager_fetch: bool = True,
    ):
        """
        Construct a new ``ExperimentAxisQueryIterable``, suitable for use with :class:`torch.utils.data.DataLoader`.

        The resulting iterator will produce a tuple containing associated slices of ``X`` and ``obs`` data, as
        a NumPy :class:`numpy.ndarray` (or optionally, :class:`scipy.sparse.csr_matrix`) and a Pandas
        :class:`pandas.DataFrame`, respectively.

        Args:
            query:
                A :class:`tiledbsoma.ExperimentAxisQuery`, defining the data to iterate over.
            X_name:
                The name of the X layer to read.
            obs_column_names:
                The names of the ``obs`` columns to return. At least one column name must be specified.
                Default is ``('soma_joinid',)``.
            batch_size:
                The number of rows of ``X`` and ``obs`` data to return in each iteration. Defaults to ``1``. A value of
                ``1`` will result in :class:`torch.Tensor` of rank 1 being returned (a single row); larger values will
                result in :class:`torch.Tensor`s of rank 2 (multiple rows). Note that a ``batch_size`` of 1 allows
                this ``IterableDataset`` to be used with :class:`torch.utils.data.DataLoader` batching, but higher
                performance can be achieved by performing batching in this class, and setting the ``DataLoader``'s
                ``batch_size`` parameter to ``None``.
            io_batch_size:
                The number of ``obs``/``X`` rows to retrieve when reading data from SOMA. This impacts
                maximum memory utilization, larger values provide better read performance, but require more memory.
            return_sparse_X:
                If ``True``, will return the ``X`` data as a :class:`scipy.sparse.csr_matrix`. If ``False`` (the
                default), will return ``X`` data as a :class:`numpy.ndarray`.
            use_eager_fetch:
                Fetch the next SOMA chunk of ``obs`` and ``X`` data immediately after a previously fetched SOMA chunk is
                made available for processing via the iterator. This allows network (or filesystem) requests to be made
                in parallel with client-side processing of the SOMA data, potentially improving overall performance at
                the cost of doubling memory utilization. Defaults to ``True``.

        Raises:
            ``ValueError`` on various unsupported or malformed parameter values.

        Lifecycle:
            experimental

        """

        super().__init__()

        # Anything set in the instance needs to be pickle-able for multi-process DataLoaders
        self.experiment_locator = _ExperimentLocator.create(query.experiment)
        self.layer_name = X_name
        self.measurement_name = query.measurement_name
        self.obs_query = query._matrix_axis_query.obs
        self.var_query = query._matrix_axis_query.var
        self.obs_column_names = list(obs_column_names)
        self.batch_size = batch_size
        self.io_batch_size = io_batch_size
        self.return_sparse_X = return_sparse_X
        self.use_eager_fetch = use_eager_fetch
        self._obs_joinids: npt.NDArray[np.int64] | None = None
        self._var_joinids: npt.NDArray[np.int64] | None = None
        self._initialized = False

        if not self.obs_column_names:
            raise ValueError("Must specify at least one value in `obs_column_names`")

    def _create_obs_joinids_partition(self) -> Iterator[npt.NDArray[np.int64]]:
        """Create iterator over obs id chunks with split size of (roughly) io_batch_size.

        As appropriate, will partition per worker.

        IMPORTANT: in any scenario using torch.distributed, where WORLD_SIZE > 1, this will
        always partition such that each process has the same number of samples. Where
        the number of obs_joinids is not evenly divisible by the number of processes,
        the number of joinids will be dropped (dropped ids can never exceed WORLD_SIZE-1).

        Abstractly, the steps taken:
        1. Split the joinids into WORLD_SIZE sections (aka number of GPUS in DDP)
        2. Trim the splits to be of equal length
        3. Partition by number of data loader workers (to not generate redundant batches
           in cases where the DataLoader is running with `n_workers>1`).

        Private method.
        """
        assert self._obs_joinids is not None
        obs_joinids: npt.NDArray[np.int64] = self._obs_joinids

        # 1. Get the split for the model replica/GPU
        world_size, rank = _get_distributed_world_rank()
        _gpu_splits = _splits(len(obs_joinids), world_size)
        _gpu_split = obs_joinids[_gpu_splits[rank] : _gpu_splits[rank + 1]]

        # 2. Trim to be all of equal length - equivalent to a "drop_last"
        # TODO: may need to add an option to do padding as well.
        min_len = np.diff(_gpu_splits).min()
        assert 0 <= (np.diff(_gpu_splits).min() - min_len) <= 1
        _gpu_split = _gpu_split[:min_len]

        obs_joinids_chunked = np.array_split(
            _gpu_split, max(1, ceil(len(_gpu_split) / self.io_batch_size))
        )

        # 3. Partition by DataLoader worker
        n_workers, worker_id = _get_worker_world_rank()
        obs_splits = _splits(len(obs_joinids_chunked), n_workers)
        obs_partition_joinids = obs_joinids_chunked[
            obs_splits[worker_id] : obs_splits[worker_id + 1]
        ].copy()

        if logger.isEnabledFor(logging.DEBUG):
            partition_size = sum([len(chunk) for chunk in obs_partition_joinids])
            logger.debug(
                f"Process {os.getpid()} {rank=}, {world_size=}, {worker_id=}, n_workers={n_workers}, {partition_size=}"
            )

        return iter(obs_partition_joinids)

    def _init_once(self, exp: soma.Experiment | None = None) -> None:
        """One-time per worker initialization.

        All operations should be idempotent in order to support pipe reset().

        Private method.
        """
        if self._initialized:
            return

        logger.debug("Initializing ExperimentAxisQueryIterable")

        if exp is None:
            # If no user-provided Experiment, open/close it ourselves
            exp_cm: ContextManager[soma.Experiment] = (
                self.experiment_locator.open_experiment()
            )
        else:
            # else, it is caller responsibility to open/close the experiment
            exp_cm = contextlib.nullcontext(exp)

        with exp_cm as exp:
            with exp.axis_query(
                measurement_name=self.measurement_name,
                obs_query=self.obs_query,
                var_query=self.var_query,
            ) as query:
                self._obs_joinids = query.obs_joinids().to_numpy()
                self._var_joinids = query.var_joinids().to_numpy()

        self._initialized = True

    def __iter__(self) -> Iterator[XObsDatum]:
        """Create iterator over query.

        Returns:
            ``iterator``

        Lifecycle:
            experimental
        """

        if (
            self.return_sparse_X
            and torch.utils.data.get_worker_info()
            and torch.utils.data.get_worker_info().num_workers > 0
        ):
            raise NotImplementedError(
                "torch does not work with sparse tensors in multi-processing mode "
                "(see https://github.com/pytorch/pytorch/issues/20248)"
            )

        world_size, rank = _get_distributed_world_rank()
        n_workers, worker_id = _get_worker_world_rank()
        logger.debug(
            f"Iterator created {rank=}, {world_size=}, {worker_id=}, {n_workers=}"
        )

        with self.experiment_locator.open_experiment() as exp:
            self._init_once(exp)
            X = exp.ms[self.measurement_name].X[self.layer_name]
            if not isinstance(X, soma.SparseNDArray):
                raise NotImplementedError(
                    "ExperimentAxisQueryIterable only supports X layers which are of type SparseNDArray"
                )

            obs_joinid_iter = self._create_obs_joinids_partition()
            _mini_batch_iter = self._mini_batch_iter(exp.obs, X, obs_joinid_iter)
            if self.use_eager_fetch:
                _mini_batch_iter = _EagerIterator(
                    _mini_batch_iter, pool=exp.context.threadpool
                )

            yield from _mini_batch_iter

    def __len__(self) -> int:
        """Return the number of batches this iterable will produce. If run in the context of :class:`torch.distributed`
        or as a multi-process loader (i.e., :class:`torch.utils.data.DataLoader` instantiated with num_workers > 0), the
        batch count will reflect the size of the data partition assigned to the active process.

        See important caveats in the PyTorch
        [:class:`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
        documentation regarding ``len(dataloader)``, which also apply to this class.

        Returns:
            ``int`` (Number of batches).

        Lifecycle:
            experimental
        """
        return self.shape[0]

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the number of batches and features that will be yielded from this :class:`tiledbsoma_ml.ExperimentAxisQueryIterable`.

        If used in multiprocessing mode (i.e. :class:`torch.utils.data.DataLoader` instantiated with num_workers > 0),
        the number of batches will reflect the size of the data partition assigned to the active process.

        Returns:
            A tuple of two ``int`` values: number of batches, number of vars.

        Lifecycle:
            experimental
        """
        self._init_once()
        assert self._obs_joinids is not None
        assert self._var_joinids is not None
        world_size, rank = _get_distributed_world_rank()
        n_workers, worker_id = _get_worker_world_rank()
        # Every "distributed" process must receive the same number of "obs" rows; the last ≤world_size may be dropped
        # (see _create_obs_joinids_partition).
        obs_per_proc = len(self._obs_joinids) // world_size
        obs_per_worker, obs_rem = divmod(obs_per_proc, n_workers)
        # obs rows assigned to this worker process
        n_worker_obs = obs_per_worker + bool(worker_id < obs_rem)
        n_batches, rem = divmod(n_worker_obs, self.batch_size)
        # (num batches this worker will produce, num features)
        return n_batches + bool(rem), len(self._var_joinids)

    def __getitem__(self, index: int) -> XObsDatum:
        raise NotImplementedError(
            "``ExperimentAxisQueryIterable can only be iterated - does not support mapping"
        )

    def _io_batch_iter(
        self,
        obs: soma.DataFrame,
        X: soma.SparseNDArray,
        obs_joinid_iter: Iterator[npt.NDArray[np.int64]],
    ) -> Iterator[Tuple[sparse.csr_matrix, pd.DataFrame]]:
        """Iterate over IO batches, i.e., SOMA query reads, producing tuples of ``(X: csr_array, obs: DataFrame)``.

        ``obs`` joinids read are controlled by the ``obs_joinid_iter``. Iterator results will be reindexed.

        Private method.
        """
        assert self._var_joinids is not None

        obs_column_names = (
            list(self.obs_column_names)
            if "soma_joinid" in self.obs_column_names
            else ["soma_joinid", *self.obs_column_names]
        )
        var_indexer = soma.IntIndexer(self._var_joinids, context=X.context)

        for obs_coords in obs_joinid_iter:
            st_time = time.perf_counter()
            obs_indexer = soma.IntIndexer(obs_coords, context=X.context)
            logger.debug(
                f"Retrieving next SOMA IO batch of length {len(obs_coords)}..."
            )

            X_tbl = X.read(coords=(obs_coords, self._var_joinids)).tables().concat()
            X_io_batch = sparse.csr_matrix(
                (
                    X_tbl["soma_data"].to_numpy(),
                    (
                        obs_indexer.get_indexer(X_tbl["soma_dim_0"]),
                        var_indexer.get_indexer(X_tbl["soma_dim_1"]),
                    ),
                ),
                shape=(len(obs_coords), len(self._var_joinids)),
            )

            # Now that X read is potentially in progress (in eager mode), go fetch obs data
            # fmt: off
            obs_io_batch = (
                obs.read(coords=(obs_coords,), column_names=obs_column_names)
                .concat()
                .to_pandas()
                .set_index("soma_joinid")
                .reindex(obs_coords, copy=False)
                .reset_index()  # demote "soma_joinid" to a column
                [self.obs_column_names]
            )  # fmt: on

            del obs_indexer, obs_coords, X_tbl
            gc.collect()

            tm = time.perf_counter() - st_time
            logger.debug(
                f"Retrieved SOMA IO batch, took {tm:.2f}sec, {X_io_batch.shape[0]/tm:0.1f} samples/sec"
            )
            yield X_io_batch, obs_io_batch

    def _mini_batch_iter(
        self,
        obs: soma.DataFrame,
        X: soma.SparseNDArray,
        obs_joinid_iter: Iterator[npt.NDArray[np.int64]],
    ) -> Iterator[XObsDatum]:
        """Break IO batches into mini-batch-sized chunks.

        Private method.
        """
        assert self._obs_joinids is not None
        assert self._var_joinids is not None

        io_batch_iter = self._io_batch_iter(obs, X, obs_joinid_iter)
        if self.use_eager_fetch:
            io_batch_iter = _EagerIterator(io_batch_iter, pool=X.context.threadpool)

        mini_batch_size = self.batch_size
        result: Tuple[NDArrayNumber, pd.DataFrame] | None = None
        for X_io_batch, obs_io_batch in io_batch_iter:
            assert X_io_batch.shape[0] == obs_io_batch.shape[0]
            assert X_io_batch.shape[1] == len(self._var_joinids)
            iob_idx = 0  # current offset into io batch
            iob_len = X_io_batch.shape[0]

            while iob_idx < iob_len:
                if result is None:
                    X_datum = (
                        X_io_batch[iob_idx : iob_idx + mini_batch_size]
                        if self.return_sparse_X
                        else X_io_batch[iob_idx : iob_idx + mini_batch_size].toarray()
                    )
                    result = (
                        X_datum,
                        obs_io_batch.iloc[
                            iob_idx : iob_idx + mini_batch_size
                        ].reset_index(drop=True),
                    )
                    iob_idx += len(result[1])
                else:
                    # Use any remnant from previous IO batch
                    to_take = min(mini_batch_size - len(result[1]), iob_len - iob_idx)
                    X_datum = (
                        sparse.vstack([result[0], X_io_batch[0:to_take]])
                        if self.return_sparse_X
                        else np.concatenate(
                            [result[0], X_io_batch[0:to_take].toarray()]
                        )
                    )
                    result = (
                        X_datum,
                        pd.concat(
                            [result[1], obs_io_batch.iloc[0:to_take]],
                            # Index `obs_batch` from 0 to N-1, instead of disjoint, concatenated pieces of IO batches'
                            # indices
                            ignore_index=True,
                        ),
                    )
                    iob_idx += to_take

                assert result[0].shape[0] == result[1].shape[0]
                if result[0].shape[0] == mini_batch_size:
                    yield result
                    result = None

        else:
            # yield the remnant, if any
            if result is not None:
                yield result


class ExperimentAxisQueryIterDataPipe(
    torchdata.datapipes.iter.IterDataPipe[  # type:ignore[misc]
        torch.utils.data.dataset.Dataset[XObsDatum]
    ],
):
    """A :class:`torchdata.datapipes.iter.IterDataPipe` implementation that loads from a :class:`tiledbsoma.SOMAExperiment`.

    This class is based upon the now-deprecated :class:`torchdata.datapipes` API, and should only be used for
    legacy code. See [GitHub issue #1196](https://github.com/pytorch/data/issues/1196) and the
    TorchData [README](https://github.com/pytorch/data/blob/v0.8.0/README.md) for more information.

    See :class:`tiledbsoma_ml.ExperimentAxisQueryIterableDataset` for more information on using this class.

    Lifecycle:
        deprecated
    """

    def __init__(
        self,
        query: soma.ExperimentAxisQuery,
        X_name: str = "raw",
        obs_column_names: Sequence[str] = ("soma_joinid",),
        batch_size: int = 1,
        io_batch_size: int = 2**16,
        return_sparse_X: bool = False,
        use_eager_fetch: bool = True,
    ):
        """
        See :class:`tiledbsoma_ml.ExperimentAxisQueryIterableDataset` for more information on using this class.

        Lifecycle:
            deprecated
        """
        super().__init__()
        self._exp_iter = ExperimentAxisQueryIterable(
            query=query,
            X_name=X_name,
            obs_column_names=obs_column_names,
            batch_size=batch_size,
            io_batch_size=io_batch_size,
            return_sparse_X=return_sparse_X,
            use_eager_fetch=use_eager_fetch,
        )

    def __iter__(self) -> Iterator[XObsDatum]:
        """
        See :class:`tiledbsoma_ml.ExperimentAxisQueryIterableDataset` for more information on using this class.

        Lifecycle:
            deprecated
        """
        batch_size = self._exp_iter.batch_size
        for X, obs in self._exp_iter:
            if batch_size == 1:
                X = X[0]  # This is a no-op for `csr_matrix`s
            yield X, obs

    def __len__(self) -> int:
        """
        See :class:`tiledbsoma_ml.ExperimentAxisQueryIterableDataset` for more information on using this class.

        Lifecycle:
            deprecated
        """
        return len(self._exp_iter)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        See :class:`tiledbsoma_ml.ExperimentAxisQueryIterableDataset` for more information on using this class.

        Lifecycle:
            deprecated
        """
        return self._exp_iter.shape


class ExperimentAxisQueryIterableDataset(
    torch.utils.data.IterableDataset[XObsDatum]  # type:ignore[misc]
):
    """A :class:`torch.utils.data.IterableDataset` implementation that loads from a :class:`tiledbsoma.SOMAExperiment`.

    This class works seamlessly with :class:`torch.utils.data.DataLoader` to load ``obs`` and ``X`` data as
    specified by a SOMA :class:`tiledbsoma.ExperimentAxisQuery`, providing an iterator over batches of
    ``obs`` and ``X`` data. Each iteration will yield a tuple containing an :class:`numpy.ndarray`
    and a :class:`pandas.DataFrame`.

    For example:

    >>> import torch
    >>> import tiledbsoma
    >>> import tiledbsoma_ml
    >>> with tiledbsoma.Experiment.open("my_experiment_path") as exp:
    ...     with exp.axis_query(measurement_name="RNA", obs_query=tiledbsoma.AxisQuery(value_filter="tissue_type=='lung'")) as query:
    ...         ds = tiledbsoma_ml.ExperimentAxisQueryIterableDataset(query)
    ...         dataloader = torch.utils.data.DataLoader(ds)
    >>> data = next(iter(dataloader))
    >>> data
    (array([0., 0., 0., ..., 0., 0., 0.], dtype=float32),
    soma_joinid
    0     57905025)
    >>> data[0]
    array([0., 0., 0., ..., 0., 0., 0.], dtype=float32)
    >>> data[1]
    soma_joinid
    0     57905025

    The ``batch_size`` parameter controls the number of rows of ``obs`` and ``X`` data that are returned in each
    iteration. If the ``batch_size`` is 1, then each result will have rank 1, else it will have rank 2. A ``batch_size``
    of 1 is compatible with :class:`torch.utils.data.DataLoader`-implemented batching, but it will usually be more
    performant to create mini-batches using this class, and set the ``DataLoader`` batch size to `None`.

    The ``obs_column_names`` parameter determines the data columns that are returned in the ``obs`` DataFrame (the
    default is a single column, containing the ``soma_joinid`` for the ``obs`` dimension).

    The ``io_batch_size`` parameter determines the number of rows read, from which mini-batches are yielded. A
    larger value will increase total memory usage and may reduce average read time per row.

    This class will detect when run in a multiprocessing mode, including multi-worker :class:`torch.utils.data.DataLoader`
    and multi-process training such as :class:`torch.nn.parallel.DistributedDataParallel`, and will automatically partition
    data appropriately. In the case of distributed training, sample partitions across all processes must be equal. Any
    data tail will be dropped.

    Lifecycle:
        experimental
    """

    def __init__(
        self,
        query: soma.ExperimentAxisQuery,
        X_name: str = "raw",
        obs_column_names: Sequence[str] = ("soma_joinid",),
        batch_size: int = 1,
        io_batch_size: int = 2**16,
        return_sparse_X: bool = False,
        use_eager_fetch: bool = True,
    ):
        """
        Construct a new ``ExperimentAxisQueryIterable``, suitable for use with :class:`torch.utils.data.DataLoader`.

        The resulting iterator will produce a tuple containing associated slices of ``X`` and ``obs`` data, as
        a NumPy ``ndarray`` (or optionally, :class:`scipy.sparse.csr_matrix`) and a Pandas ``DataFrame`` respectively.

        Args:
            query:
                A :class:`tiledbsoma.ExperimentAxisQuery`, defining the data which will be iterated over.
            X_name:
                The name of the ``X`` layer to read.
            obs_column_names:
                The names of the ``obs`` columns to return. At least one column name must be specified.
                Default is ``('soma_joinid',)``.
            batch_size:
                The number of rows of ``X`` and ``obs`` data to return in each iteration. Defaults to ``1``. A value of
                ``1`` will result in :class:`torch.Tensor` of rank 1 being returned (a single row); larger values will
                result in :class:`torch.Tensor`\ s of rank 2 (multiple rows).

                Note that a ``batch_size`` of 1 allows this ``IterableDataset`` to be used with :class:`torch.utils.data.DataLoader`
                batching, but you will achieve higher performance by performing batching in this class, and setting the ``DataLoader``
                batch_size parameter to ``None``.
            io_batch_size:
                The number of ``obs``/``X`` rows to retrieve when reading data from SOMA.
            return_sparse_X:
                If ``True``, will return the ``X`` data as a :class:`scipy.sparse.csr_matrix`. If ``False`` (the default), will
                return ``X`` data as a :class:`numpy.ndarray`.
            use_eager_fetch:
                Fetch the next SOMA chunk of ``obs`` and ``X`` data immediately after a previously fetched SOMA chunk is made
                available for processing via the iterator. This allows network (or filesystem) requests to be made in
                parallel with client-side processing of the SOMA data, potentially improving overall performance at the
                cost of doubling memory utilization. Defaults to ``True``.

        Raises:
            ``ValueError`` on various unsupported or malformed parameter values.

        Lifecycle:
            experimental

        """
        super().__init__()
        self._exp_iter = ExperimentAxisQueryIterable(
            query=query,
            X_name=X_name,
            obs_column_names=obs_column_names,
            batch_size=batch_size,
            io_batch_size=io_batch_size,
            return_sparse_X=return_sparse_X,
            use_eager_fetch=use_eager_fetch,
        )

    def __iter__(self) -> Iterator[XObsDatum]:
        """Create ``Iterator`` yielding "mini-batch" tuples of :class:`numpy.ndarray` (or :class:`scipy.csr_matrix`) and
        :class:`pandas.DataFrame`.

        Returns:
            ``iterator``

        Lifecycle:
            experimental
        """
        batch_size = self._exp_iter.batch_size
        for X, obs in self._exp_iter:
            if batch_size == 1:
                X = X[0]  # This is a no-op for `csr_matrix`s
            yield X, obs

    def __len__(self) -> int:
        """Return number of batches this iterable will produce.

        See important caveats in the PyTorch
        [:class:`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
        documentation regarding ``len(dataloader)``, which also apply to this class.

        Returns:
            ``int`` (number of batches).

        Lifecycle:
            experimental
        """
        return len(self._exp_iter)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the number of batches and features that will be yielded from this :class:`tiledbsoma_ml.ExperimentAxisQueryIterable`.

        If used in multiprocessing mode (i.e. :class:`torch.utils.data.DataLoader` instantiated with num_workers > 0),
        the number of batches will reflect the size of the data partition assigned to the active process.

        Returns:
            A tuple of two ``int`` values: number of batches, number of vars.

        Lifecycle:
            experimental
        """
        return self._exp_iter.shape


UNSUPPORTED_DATALOADER_ARGS = {
    "shuffle",
    "batch_size",
    "sampler",
    "batch_sampler",
}


def experiment_dataloader(
    ds: torchdata.datapipes.iter.IterDataPipe | torch.utils.data.IterableDataset,
    **dataloader_kwargs: Any,
) -> torch.utils.data.DataLoader:
    """Factory method for :class:`torch.utils.data.DataLoader`. This method can be used to safely instantiate a
    :class:`torch.utils.data.DataLoader` that works with :class:`tiledbsoma_ml.ExperimentAxisQueryIterableDataset`
    or :class:`tiledbsoma_ml.ExperimentAxisQueryIterDataPipe`.

    Several :class:`torch.utils.data.DataLoader` constructor parameters are not applicable, or are non-performant,
    when using loaders from this module, including ``shuffle``, ``batch_size``, ``sampler``, and ``batch_sampler``.
    Specifying any of these parameters will result in an error.

    Refer to ``https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader`` for more information on
    :class:`torch.utils.data.DataLoader` parameters.

    Args:
        ds:
            A :class:`torch.utils.data.IterableDataset` or a :class:`torchdata.datapipes.iter.IterDataPipe`. May
            include chained data pipes.
        **dataloader_kwargs:
            Additional keyword arguments to pass to the :class:`torch.utils.data.DataLoader` constructor,
            except for ``shuffle``, ``batch_size``, ``sampler``, and ``batch_sampler``, which are not
            supported when using data loaders in this module.

    Returns:
        A :class:`torch.utils.data.DataLoader`.

    Raises:
        ValueError: if any of the ``shuffle``, ``batch_size``, ``sampler``, or ``batch_sampler`` params
            are passed as keyword arguments.

    Lifecycle:
        experimental
    """
    if unsupported_dataloader_args := UNSUPPORTED_DATALOADER_ARGS.intersection(
        dataloader_kwargs
    ):
        raise ValueError(
            f"The {','.join(unsupported_dataloader_args)} DataLoader parameters are not supported"
        )

    if dataloader_kwargs.get("num_workers", 0) > 0:
        _init_multiprocessing()

    if "collate_fn" not in dataloader_kwargs:
        # PyTorch's default collate_fn manipulates batches, which we don't want; replace it with a no-op.
        dataloader_kwargs["collate_fn"] = _collate_noop

    return torch.utils.data.DataLoader(
        ds,
        batch_size=None,  # batching is handled by upstream iterator
        shuffle=False,  # shuffling is handled by upstream iterator
        **dataloader_kwargs,
    )


def _collate_noop(datum: _T) -> _T:
    """Noop collation for use with a dataloader instance.

    Private.
    """
    return datum


def _splits(total_length: int, sections: int) -> npt.NDArray[np.intp]:
    """For ``total_length`` points, compute start/stop offsets that split the length into roughly equal sizes.

    A total_length of L, split into N sections, will return L%N sections of size L//N+1,
    and the remainder as size L//N. This results in the same split as numpy.array_split,
    for an array of length L and sections N.

    Private.

    Examples
    --------
    >>> _splits(10, 3)
    array([0,  4,  7, 10])
    >>> _splits(4, 2)
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
    _batched = itertools.batched
else:

    def _batched(iterable: Iterable[_T_co], n: int) -> Iterator[Tuple[_T_co, ...]]:
        """Same as the Python 3.12+ ``itertools.batched`` -- polyfill for old Python versions."""
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch


def _get_distributed_world_rank() -> Tuple[int, int]:
    """Return tuple containing equivalent of ``torch.distributed`` world size and rank."""
    world_size, rank = 1, 0
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
    elif "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Lightning doesn't use RANK! LOCAL_RANK is only for the local node. There
        # is a NODE_RANK for the node's rank, but no way to tell the local node's
        # world. So computing a global rank is impossible(?).  Using LOCAL_RANK as a
        # proxy, which works fine on a single-CPU box. TODO: could throw/error
        # if NODE_RANK != 0.
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["LOCAL_RANK"])
    elif torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    return world_size, rank


def _get_worker_world_rank() -> Tuple[int, int]:
    """Return number of DataLoader workers and our worker rank/id"""
    num_workers, worker = 1, 0
    if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
        num_workers = int(os.environ["NUM_WORKERS"])
        worker = int(os.environ["WORKER"])
    else:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker = worker_info.id
    return num_workers, worker


def _init_multiprocessing() -> None:
    """Ensures use of "spawn" for starting child processes with multiprocessing.

    Forked processes are known to be problematic:
      https://pytorch.org/docs/stable/notes/multiprocessing.html#avoiding-and-fighting-deadlocks
    Also, CUDA does not support forked child processes:
      https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing

    Private.
    """
    orig_start_method = torch.multiprocessing.get_start_method()
    if orig_start_method != "spawn":
        if orig_start_method:
            logger.warning(
                "switching torch multiprocessing start method from "
                f'"{torch.multiprocessing.get_start_method()}" to "spawn"'
            )
        torch.multiprocessing.set_start_method("spawn", force=True)
