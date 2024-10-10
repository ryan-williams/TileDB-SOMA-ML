# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

import logging
from typing import Any, TypeVar

import torch
from torch.utils.data import DataLoader, IterableDataset
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger("tiledbsoma_ml.dataloader")

_T = TypeVar("_T")

UNSUPPORTED_DATALOADER_ARGS = {
    "shuffle",
    "batch_size",
    "sampler",
    "batch_sampler",
}


def experiment_dataloader(
    ds: IterDataPipe | IterableDataset,
    **dataloader_kwargs: Any,
) -> DataLoader:
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

    if "collate_fn" not in dataloader_kwargs:
        # PyTorch's default collate_fn manipulates batches, which we don't want; replace it with a no-op.
        dataloader_kwargs["collate_fn"] = _collate_noop

    return DataLoader(
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
